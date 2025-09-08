# speech_pipeline_manager.py (Corrected and Reverted to stable version)
from typing import Optional, Callable
import threading
import logging
import time
from queue import Queue, Empty
import sys

from audio_module import AudioProcessor
from text_similarity import TextSimilarity
from text_context import TextContext
from llm_module import LLM
from colors import Colors

logger = logging.getLogger(__name__)

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    logger.info("🗣️📄 System prompt loaded from file.")
except FileNotFoundError:
    logger.warning("🗣️📄 system_prompt.txt not found. Using default system prompt.")
    system_prompt = "You are a helpful assistant."


USE_ORPHEUS_UNCENSORED = False

orpheus_prompt_addon_normal = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", and " <gasp> ".

Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon_uncensored = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <moans> ", " <panting> ", " <grunting> ", " <gagging sounds> ", " <chokeing> ", " <kissing noises> ", " <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", " <gasp> ".
Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon = orpheus_prompt_addon_uncensored if USE_ORPHEUS_UNCENSORED else orpheus_prompt_addon_normal


class PipelineRequest:
    def __init__(self, action: str, data: Optional[any] = None):
        self.action = action
        self.data = data
        self.timestamp = time.time()

class RunningGeneration:
    def __init__(self, id: int):
        self.id: int = id
        self.text: Optional[str] = None
        self.timestamp = time.time()
        self.llm_generator = None
        self.llm_finished: bool = False
        self.llm_finished_event = threading.Event()
        self.llm_aborted: bool = False
        self.quick_answer: str = ""
        self.quick_answer_provided: bool = False
        self.quick_answer_first_chunk_ready: bool = False
        self.quick_answer_overhang: str = ""
        self.tts_quick_started: bool = False
        self.tts_quick_allowed_event = threading.Event()
        self.audio_chunks = Queue()
        self.audio_quick_finished: bool = False
        self.audio_quick_aborted: bool = False
        self.tts_quick_finished_event = threading.Event()
        self.abortion_started: bool = False
        self.tts_final_finished_event = threading.Event()
        self.tts_final_started: bool = False
        self.audio_final_aborted: bool = False
        self.audio_final_finished: bool = False
        self.final_answer: str = ""
        self.completed: bool = False


class SpeechPipelineManager:
    def __init__(
            self,
            tts_engine: str = "kokoro",
            llm_provider: str = "ollama",
            llm_model: str = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M",
            no_think: bool = False,
            orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf",
        ):
        self.tts_engine = tts_engine
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.no_think = no_think
        self.orpheus_model = orpheus_model

        self.system_prompt = system_prompt
        if tts_engine == "orpheus":
            self.system_prompt += f"\n{orpheus_prompt_addon}"

        # --- Instance Dependencies (ORIGINAL ORDER) ---
        self.audio = AudioProcessor(
            engine=self.tts_engine,
            orpheus_model=self.orpheus_model
        )
        self.audio.on_first_audio_chunk_synthesize = self.on_first_audio_chunk_synthesize
        self.text_similarity = TextSimilarity(focus='end', n_words=5)
        self.text_context = TextContext()
        self.generation_counter: int = 0
        self.abort_lock = threading.Lock()
        self.llm = LLM(
            backend=self.llm_provider,
            model=self.llm_model,
            system_prompt=self.system_prompt,
            no_think=no_think,
        )
        self.llm.prewarm()
        self.llm_inference_time = self.llm.measure_inference_time()
        logger.debug(f"🗣️🧠🕒 LLM inference time: {self.llm_inference_time:.2f}ms")

        # --- State ---
        self.history = []
        self.requests_queue = Queue()
        self.running_generation: Optional[RunningGeneration] = None

        # --- Threading Events ---
        self.shutdown_event = threading.Event()
        self.generator_ready_event = threading.Event()
        self.llm_answer_ready_event = threading.Event()
        self.stop_everything_event = threading.Event()
        self.stop_llm_request_event = threading.Event()
        self.stop_llm_finished_event = threading.Event()
        self.stop_tts_quick_request_event = threading.Event()
        self.stop_tts_quick_finished_event = threading.Event()
        self.stop_tts_final_request_event = threading.Event()
        self.stop_tts_final_finished_event = threading.Event()
        self.abort_completed_event = threading.Event()
        self.abort_block_event = threading.Event()
        self.abort_block_event.set()
        self.check_abort_lock = threading.Lock()

        # --- State Flags ---
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.previous_request = None

        # --- Worker Threads ---
        self.request_processing_thread = threading.Thread(target=self._request_processing_worker, name="RequestProcessingThread", daemon=True)
        self.llm_inference_thread = threading.Thread(target=self._llm_inference_worker, name="LLMProcessingThread", daemon=True)
        self.tts_quick_inference_thread = threading.Thread(target=self._tts_quick_inference_worker, name="TTSQuickProcessingThread", daemon=True)
        self.tts_final_inference_thread = threading.Thread(target=self._tts_final_inference_worker, name="TTSFinalProcessingThread", daemon=True)

        self.request_processing_thread.start()
        self.llm_inference_thread.start()
        self.tts_quick_inference_thread.start()
        self.tts_final_inference_thread.start()

        self.on_partial_assistant_text: Optional[Callable[[str], None]] = None

        self.full_output_pipeline_latency = self.llm_inference_time + self.audio.tts_inference_time
        logger.info(f"🗣️⏱️ Full output pipeline latency: {self.full_output_pipeline_latency:.2f}ms (LLM: {self.llm_inference_time:.2f}ms, TTS: {self.audio.tts_inference_time:.2f}ms)")

        logger.info("🗣️🚀 SpeechPipelineManager initialized and workers started.")

    def is_valid_gen(self) -> bool:
        return self.running_generation is not None and not self.running_generation.abortion_started

    def _request_processing_worker(self):
        logger.info("🗣️🚀 Request Processor: Starting...")
        while not self.shutdown_event.is_set():
            try:
                request = self.requests_queue.get(block=True, timeout=1)

                if self.previous_request:
                    if self.previous_request.data == request.data and isinstance(request.data, str):
                        if request.timestamp - self.previous_request.timestamp < 2:
                            logger.info(f"🗣️🗑️ Request Processor: Skipping duplicate request - {request.action}")
                            continue

                while not self.requests_queue.empty():
                    skipped_request = self.requests_queue.get(False)
                    logger.debug(f"🗣️🗑️ Request Processor: Skipping older request - {skipped_request.action}")
                    request = skipped_request
                
                self.abort_block_event.wait()
                logger.debug(f"🗣️🔄 Request Processor: Processing most recent request - {request.action}")
                
                if request.action == "prepare":
                    self.process_prepare_generation(request.data)
                    self.previous_request = request
                elif request.action == "finish":
                     logger.info(f"🗣️🤷 Request Processor: Received 'finish' action (currently no-op).")
                     self.previous_request = request
                else:
                    logger.warning(f"🗣️❓ Request Processor: Unknown action '{request.action}'")

            except Empty:
                continue
            except Exception as e:
                logger.exception(f"🗣️💥 Request Processor: Error: {e}")
        logger.info("🗣️🏁 Request Processor: Shutting down.")

    def on_first_audio_chunk_synthesize(self):
        logger.info("🗣️🎶 First audio chunk synthesized. Setting TTS quick allowed event.")
        if self.running_generation:
            self.running_generation.quick_answer_first_chunk_ready = True

    def preprocess_chunk(self, chunk: str) -> str:
        return chunk.replace("—", "-").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("…", "...")

    def clean_quick_answer(self, text: str) -> str:
        patterns_to_remove = ["<think>", "</think>", "\n", " "]
        previous_text = None
        current_text = text
        
        while previous_text != current_text:
            previous_text = current_text
            
            for pattern in patterns_to_remove:
                while current_text.startswith(pattern):
                    current_text = current_text[len(pattern):]
        
        return current_text

    def _llm_inference_worker(self):
        logger.info("🗣️🧠 LLM Worker: Starting...")
        while not self.shutdown_event.is_set():
            
            ready = self.generator_ready_event.wait(timeout=1.0)
            if not ready:
                continue

            if self.stop_llm_request_event.is_set():
                logger.info("🗣️🧠❌ LLM Worker: Abort detected while waiting for generator_ready_event.")
                self.stop_llm_request_event.clear()
                self.stop_llm_finished_event.set()
                self.llm_generation_active = False
                continue

            self.generator_ready_event.clear()
            self.stop_everything_event.clear()
            current_gen = self.running_generation

            if not current_gen or not current_gen.llm_generator:
                logger.warning("🗣️🧠❓ LLM Worker: No valid generation or generator found after event.")
                self.llm_generation_active = False
                continue

            gen_id = current_gen.id
            logger.info(f"🗣️🧠🔄 [Gen {gen_id}] LLM Worker: Processing generation...")

            self.llm_generation_active = True
            self.stop_llm_finished_event.clear()
            start_time = time.time()
            token_count = 0

            try:
                for chunk in current_gen.llm_generator:
                    if self.stop_llm_request_event.is_set():
                        logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM Worker: Stop request detected during iteration.")
                        self.stop_llm_request_event.clear()
                        current_gen.llm_aborted = True
                        break

                    chunk = self.preprocess_chunk(chunk)
                    token_count += 1
                    current_gen.quick_answer += chunk
                    if self.no_think:
                        current_gen.quick_answer = self.clean_quick_answer(current_gen.quick_answer)

                    if token_count == 1:
                        logger.info(f"🗣️🧠⏱️ [Gen {gen_id}] LLM Worker: TTFT: {(time.time() - start_time):.4f}s")

                    if not current_gen.quick_answer_provided:
                        context, overhang = self.text_context.get_context(current_gen.quick_answer)
                        if context:
                            logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM Worker:  {Colors.apply('QUICK ANSWER FOUND:').magenta} {context}, overhang: {overhang}")
                            current_gen.quick_answer = context
                            if self.on_partial_assistant_text:
                                self.on_partial_assistant_text(current_gen.quick_answer)
                            current_gen.quick_answer_overhang = overhang
                            current_gen.quick_answer_provided = True
                            self.llm_answer_ready_event.set()
                            break

                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM Worker: Generator loop finished%s" % (" (Aborted)" if current_gen.llm_aborted else ""))

                if not current_gen.llm_aborted and not current_gen.quick_answer_provided:
                    logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM Worker: No context boundary found, using full response as quick answer.")
                    current_gen.quick_answer_provided = True
                    if self.on_partial_assistant_text:
                        self.on_partial_assistant_text(current_gen.quick_answer)
                    self.llm_answer_ready_event.set()

            except Exception as e:
                logger.exception(f"🗣️🧠💥 [Gen {gen_id}] LLM Worker: Error during generation: {e}")
                current_gen.llm_aborted = True
            finally:
                self.llm_generation_active = False
                self.stop_llm_finished_event.set()

                if current_gen.llm_aborted:
                    logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM Aborted, requesting TTS quick/final stop.")
                    self.stop_tts_quick_request_event.set()
                    self.stop_tts_final_request_event.set()
                    self.llm_answer_ready_event.set()

                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM Worker: Finished processing cycle.")

                current_gen.llm_finished = True
                current_gen.llm_finished_event.set()

    def check_abort(self, txt: str, wait_for_finish: bool = True, abort_reason: str = "unknown") -> bool:
        with self.check_abort_lock:
            if self.running_generation:
                current_gen_id_str = f"Gen {self.running_generation.id}"
                logger.info(f"🗣️🛑❓ {current_gen_id_str} Abort check requested (reason: {abort_reason})")

                if self.running_generation.abortion_started:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} Active generation is already aborting, waiting to finish (if requested).")

                    if wait_for_finish:
                        start_time = time.time()
                        completed = self.abort_completed_event.wait(timeout=5.0)

                        if not completed:
                             logger.error(f"🗣️🛑💥💥 {current_gen_id_str} Timeout waiting for ongoing abortion to complete. State inconsistency possible!")
                             self.running_generation = None
                        elif self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} Abortion completed event set, but running_generation still exists. State inconsistency likely!")
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} Ongoing abortion finished.")
                    else:
                        logger.info(f"🗣️🛑🏃 {current_gen_id_str} Not waiting for ongoing abortion as wait_for_finish=False")

                    return True
                else:
                    logger.info(f"🗣️🛑🤔 {current_gen_id_str} Found active generation, checking text similarity.")
                    try:
                        if self.running_generation.text is None:
                            logger.warning(f"🗣️🛑❓ {current_gen_id_str} Running generation text is None, cannot compare similarity. Assuming different.")
                            similarity = 0.0
                        else:
                            similarity = self.text_similarity.calculate_similarity(self.running_generation.text, txt)
                    except Exception as e:
                        logger.warning(f"🗣️🛑💥 {current_gen_id_str} Error calculating similarity: {e}. Assuming different.")
                        similarity = 0.0

                    if similarity >= 0.95:
                        logger.info(f"🗣️🛑🙅 {current_gen_id_str} Text ('{txt[:30]}...') too similar ({similarity:.2f}) to current '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'. Ignoring.")
                        return False

                    logger.info(f"🗣️🛑🚀 {current_gen_id_str} Text ('{txt[:30]}...') different enough ({similarity:.2f}) from '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'. Requesting synchronous abort.")
                    start_time = time.time()
                    self.abort_generation(wait_for_completion=wait_for_finish, timeout=7.0, reason=f"check_abort found different text ({abort_reason})")

                    if wait_for_finish:
                        if self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} !!! Abort call completed but running_generation is still not None. State inconsistency likely!")
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} Synchronous abort completed in {time.time() - start_time:.2f}s.")

                    return True
            else:
                logger.info("🗣️🛑🤷 No active generation found during abort check.")
                return False

    def _tts_quick_inference_worker(self):
        logger.info("🗣️👄🚀 Quick TTS Worker: Starting...")
        while not self.shutdown_event.is_set():
            ready = self.llm_answer_ready_event.wait(timeout=1.0)
            if not ready:
                continue

            if self.stop_tts_quick_request_event.is_set():
                logger.info("🗣️👄❌ Quick TTS Worker: Abort detected while waiting for llm_answer_ready_event.")
                self.stop_tts_quick_request_event.clear()
                self.stop_tts_quick_finished_event.set()
                self.tts_quick_generation_active = False
                continue

            self.llm_answer_ready_event.clear()
            current_gen = self.running_generation

            if not current_gen or not current_gen.quick_answer:
                logger.warning("🗣️👄❓ Quick TTS Worker: No valid generation or quick answer found after event.")
                self.tts_quick_generation_active = False
                continue

            if current_gen.audio_quick_aborted or current_gen.abortion_started:
                logger.info(f"🗣️👄❌ [Gen {current_gen.id}] Quick TTS Worker: Generation already marked as aborted. Skipping.")
                continue

            gen_id = current_gen.id
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Quick TTS Worker: Processing TTS for quick answer...")

            self.tts_quick_generation_active = True
            self.stop_tts_quick_finished_event.clear()
            current_gen.tts_quick_finished_event.clear()
            current_gen.tts_quick_started = True

            allowed_to_speak = True

            try:
                if self.stop_tts_quick_request_event.is_set() or current_gen.abortion_started:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Worker: Aborting TTS synthesis due to stop request or abortion flag.")
                     current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄🎶 [Gen {gen_id}] Quick TTS Worker: Synthesizing: '{current_gen.quick_answer[:50]}...'")
                    completed = self.audio.synthesize(
                        current_gen.quick_answer,
                        current_gen.audio_chunks,
                        self.stop_tts_quick_request_event
                    )

                    if not completed:
                        logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Worker: Synthesis stopped via event.")
                        current_gen.audio_quick_aborted = True
                    else:
                        logger.info(f"🗣️👄✅ [Gen {gen_id}] Quick TTS Worker: Synthesis completed successfully.")

            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Quick TTS Worker: Error during synthesis: {e}")
                current_gen.audio_quick_aborted = True
            finally:
                self.tts_quick_generation_active = False
                self.stop_tts_quick_finished_event.set()
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] Quick TTS Worker: Finished processing cycle.")

                if current_gen.audio_quick_aborted or self.stop_tts_quick_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Marked as Aborted/Incomplete.")
                    self.stop_tts_quick_request_event.clear()
                    current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Quick TTS Finished Successfully.")
                    current_gen.tts_quick_finished_event.set()

                current_gen.audio_quick_finished = True

    def _tts_final_inference_worker(self):
        logger.info("🗣️👄🚀 Final TTS Worker: Starting...")
        while not self.shutdown_event.is_set():
            current_gen = self.running_generation
            time.sleep(0.01)

            if not current_gen: continue
            if current_gen.tts_final_started: continue
            if not current_gen.tts_quick_started: continue
            if not current_gen.audio_quick_finished: continue

            gen_id = current_gen.id

            if current_gen.audio_quick_aborted:
                continue
            if not current_gen.quick_answer_provided:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] Final TTS Worker: Quick answer boundary was not found, skipping final TTS (quick TTS handled everything).")
                 continue
            if current_gen.abortion_started:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] Final TTS Worker: Generation is aborting, skipping final TTS.")
                 continue

            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Final TTS Worker: Processing final TTS...")

            def get_generator():
                if current_gen.quick_answer_overhang:
                    preprocessed_overhang = self.preprocess_chunk(current_gen.quick_answer_overhang)
                    logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Yielding overhang: '{preprocessed_overhang[:50]}...'")
                    current_gen.final_answer += preprocessed_overhang
                    if self.on_partial_assistant_text:
                         logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Worker on_partial_assistant_text: Sending overhang.")
                         try:
                            self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                         except Exception as cb_e:
                             logger.warning(f"🗣️💥 Callback error in on_partial_assistant_text (overhang): {cb_e}")
                    yield preprocessed_overhang

                logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Yielding remaining LLM chunks...")
                try:
                    for chunk in current_gen.llm_generator:
                         if self.stop_tts_final_request_event.is_set():
                             logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Gen: Stop request detected during LLM iteration.")
                             current_gen.audio_final_aborted = True
                             break

                         preprocessed_chunk = self.preprocess_chunk(chunk)
                         current_gen.final_answer += preprocessed_chunk
                         if self.on_partial_assistant_text:
                            try:
                                 self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                            except Exception as cb_e:
                                 logger.warning(f"🗣️💥 Callback error in on_partial_assistant_text (final chunk): {cb_e}")

                         yield preprocessed_chunk
                    logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Finished iterating LLM chunks.")
                except Exception as gen_e:
                     logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS Gen: Error iterating LLM generator: {gen_e}")
                     current_gen.audio_final_aborted = True

            self.tts_final_generation_active = True
            self.stop_tts_final_finished_event.clear()
            current_gen.tts_final_started = True
            current_gen.tts_final_finished_event.clear()

            try:
                logger.info(f"🗣️👄🎶 [Gen {gen_id}] Final TTS Worker: Synthesizing remaining text...")
                completed = self.audio.synthesize_generator(
                    get_generator(),
                    current_gen.audio_chunks,
                    self.stop_tts_final_request_event
                )

                if not completed:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Worker: Synthesis stopped via event.")
                     current_gen.audio_final_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Final TTS Worker: Synthesis completed successfully.")

            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS Worker: Error during synthesis: {e}")
                current_gen.audio_final_aborted = True
            finally:
                self.tts_final_generation_active = False
                self.stop_tts_final_finished_event.set()
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] Final TTS Worker: Finished processing cycle.")

                if current_gen.audio_final_aborted or self.stop_tts_final_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Marked as Aborted/Incomplete.")
                    self.stop_tts_final_request_event.clear()
                    current_gen.audio_final_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Final TTS Finished Successfully.")
                    current_gen.tts_final_finished_event.set()

                current_gen.audio_final_finished = True

    def process_prepare_generation(self, txt: str):
        id_in_spec = self.generation_counter + 1
        aborted = self.check_abort(txt, wait_for_finish=True, abort_reason=f"process_prepare_generation for new id {id_in_spec}")

        self.generation_counter += 1
        new_gen_id = self.generation_counter
        logger.info(f"🗣️✨🔄 [Gen {new_gen_id}] Preparing new generation for: '{txt[:50]}...'")

        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.llm_answer_ready_event.clear()
        self.generator_ready_event.clear()
        self.stop_llm_request_event.clear()
        self.stop_llm_finished_event.clear()
        self.stop_tts_quick_request_event.clear()
        self.stop_tts_quick_finished_event.clear()
        self.stop_tts_final_request_event.clear()
        self.stop_tts_final_finished_event.clear()
        self.abort_completed_event.clear()
        self.abort_block_event.set()

        self.running_generation = RunningGeneration(id=new_gen_id)
        self.running_generation.text = txt

        try:
            logger.info(f"🗣️🧠🚀 [Gen {new_gen_id}] Calling LLM generate...")
            self.running_generation.llm_generator = self.llm.generate(
                text=txt,
                history=self.history,
                use_system_prompt=True,
            )
            logger.info(f"🗣️🧠✔️ [Gen {new_gen_id}] LLM generator created. Setting generator ready event.")
            self.generator_ready_event.set()
        except Exception as e:
            logger.exception(f"🗣️🧠💥 [Gen {new_gen_id}] Failed to create LLM generator: {e}")
            self.running_generation = None

    def process_abort_generation(self):
        with self.abort_lock:
            current_gen_obj = self.running_generation
            current_gen_id_str = f"Gen {current_gen_obj.id}" if current_gen_obj else "Gen None"

            if current_gen_obj is None or current_gen_obj.abortion_started:
                if current_gen_obj is None:
                    logger.info(f"🗣️🛑🤷 {current_gen_id_str} No active generation found to abort.")
                else:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} Abortion already in progress.")
                self.abort_completed_event.set()
                self.abort_block_event.set()
                return

            logger.info(f"🗣️🛑🚀 {current_gen_id_str} Abortion process starting...")
            current_gen_obj.abortion_started = True
            self.abort_block_event.clear()
            self.abort_completed_event.clear()
            self.stop_everything_event.set()
            aborted_something = False

            is_llm_potentially_active = self.llm_generation_active or self.generator_ready_event.is_set()
            if is_llm_potentially_active:
                logger.info(f"🗣️🛑🧠❌ {current_gen_id_str} - Stopping LLM...")
                self.stop_llm_request_event.set()
                self.generator_ready_event.set()
                stopped = self.stop_llm_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑🧠👍 {current_gen_id_str} LLM stopped confirmation received.")
                    self.stop_llm_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑🧠⏱️ {current_gen_id_str} Timeout waiting for LLM stop confirmation.")
                if hasattr(self.llm, 'cancel_generation'):
                    logger.info(f"🗣️🛑🧠🔌 {current_gen_id_str} Calling external LLM cancel_generation.")
                    try:
                        self.llm.cancel_generation()
                    except Exception as cancel_e:
                         logger.warning(f"🗣️🛑🧠💥 {current_gen_id_str} Error during external LLM cancel: {cancel_e}")
                self.llm_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑🧠📴 {current_gen_id_str} LLM appears inactive, no stop needed.")
            self.stop_llm_request_event.clear()

            is_tts_quick_potentially_active = self.tts_quick_generation_active or self.llm_answer_ready_event.is_set()
            if is_tts_quick_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} Stopping Quick TTS...")
                self.stop_tts_quick_request_event.set()
                self.llm_answer_ready_event.set()
                stopped = self.stop_tts_quick_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} Quick TTS stopped confirmation received.")
                    self.stop_tts_quick_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} Timeout waiting for Quick TTS stop confirmation.")
                self.tts_quick_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} Quick TTS appears inactive, no stop needed.")
            self.stop_tts_quick_request_event.clear()

            is_tts_final_potentially_active = self.tts_final_generation_active
            if is_tts_final_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} Stopping Final TTS...")
                self.stop_tts_final_request_event.set()
                stopped = self.stop_tts_final_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} Final TTS stopped confirmation received.")
                    self.stop_tts_final_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} Timeout waiting for Final TTS stop confirmation.")
                self.tts_final_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} Final TTS appears inactive, no stop needed.")
            self.stop_tts_final_request_event.clear()

            if hasattr(self.audio, 'stop_playback'):
                logger.info(f"🗣️🛑🔊 {current_gen_id_str} Requesting audio playback stop.")
                try:
                    self.audio.stop_playback()
                except Exception as audio_e:
                    logger.warning(f"🗣️🛑🔊💥 {current_gen_id_str} Error stopping audio playback: {audio_e}")

            if self.running_generation is not None and self.running_generation.id == current_gen_obj.id:
                logger.info(f"🗣️🛑🧹 {current_gen_id_str} Clearing running generation object.")
                if current_gen_obj.llm_generator and hasattr(current_gen_obj.llm_generator, 'close'):
                    try:
                        logger.info(f"🗣️🛑🧠🔌 {current_gen_id_str} Closing LLM generator stream.")
                        current_gen_obj.llm_generator.close()
                    except Exception as e:
                        logger.warning(f"🗣️🛑🧠💥 {current_gen_id_str} Error closing LLM generator: {e}")
                self.running_generation = None
            elif self.running_generation is not None and self.running_generation.id != current_gen_obj.id:
                 logger.warning(f"🗣️🛑❓ {current_gen_id_str} Mismatch: self.running_generation changed during abort (now Gen {self.running_generation.id}). Clearing current ref.")
                 self.running_generation = None
            elif aborted_something:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} Worker(s) aborted but running_generation was already None.")
            else:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} Nothing seemed active to abort, running_generation is None.")

            self.generator_ready_event.clear()
            self.llm_answer_ready_event.clear()

            logger.info(f"🗣️🛑✅ {current_gen_id_str} Abort processing complete. Setting completion event and releasing block.")
            self.abort_completed_event.set()
            self.abort_block_event.set()

    def prepare_generation(self, txt: str):
        logger.info(f"🗣️📥 Queueing 'prepare' request for: '{txt[:50]}...'")
        self.requests_queue.put(PipelineRequest("prepare", txt))

    def finish_generation(self):
        logger.info(f"🗣️📥 Queueing 'finish' request")
        self.requests_queue.put(PipelineRequest("finish"))

    def abort_generation(self, wait_for_completion: bool = False, timeout: float = 7.0, reason: str = ""):
        if self.shutdown_event.is_set():
            logger.warning("🗣️🔌 Shutdown in progress, ignoring abort request.")
            return

        gen_id_str = f"Gen {self.running_generation.id}" if self.running_generation else "Gen None"
        logger.info(f"🗣️🛑🚀 Requesting 'abort' (wait={wait_for_completion}, reason='{reason}') for {gen_id_str}")

        self.process_abort_generation()

        if wait_for_completion:
            logger.info(f"🗣️🛑⏳ Waiting for abort completion (timeout={timeout}s)...")
            completed = self.abort_completed_event.wait(timeout=timeout)
            if completed:
                logger.info(f"🗣️🛑✅ Abort completion confirmed.")
            else:
                logger.warning(f"🗣️🛑⏱️ Timeout waiting for abort completion event.")
            self.abort_block_event.set()

    def reset(self):
        logger.info("🗣️🔄 Resetting pipeline state...")
        self.abort_generation(wait_for_completion=True, timeout=7.0, reason="reset")
        self.history = []
        logger.info("🗣️🧹 History cleared. Reset complete.")

    def shutdown(self):
        logger.info("🗣️🔌 Initiating shutdown...")
        self.shutdown_event.set()

        logger.info("🗣️🔌🛑 Attempting final abort before joining threads...")
        self.abort_generation(wait_for_completion=True, timeout=3.0, reason="shutdown")

        logger.info("🗣️🔌🔔 Signaling events to wake up any waiting threads...")
        self.generator_ready_event.set()
        self.llm_answer_ready_event.set()
        self.stop_llm_finished_event.set()
        self.stop_tts_quick_finished_event.set()
        self.stop_tts_final_finished_event.set()
        self.abort_completed_event.set()
        self.abort_block_event.set()

        threads_to_join = [
            (self.request_processing_thread, "Request Processor"),
            (self.llm_inference_thread, "LLM Worker"),
            (self.tts_quick_inference_thread, "Quick TTS Worker"),
            (self.tts_final_inference_thread, "Final TTS Worker"),
        ]

        for thread, name in threads_to_join:
             if thread.is_alive():
                 logger.info(f"🗣️🔌⏳ Joining {name}...")
                 thread.join(timeout=5.0)
                 if thread.is_alive():
                     logger.warning(f"🗣️🔌⏱️ {name} thread did not join cleanly.")
             else:
                  logger.info(f"🗣️🔌👍 {name} thread already finished.")

        logger.info("🗣️🔌✅ Shutdown complete.")
