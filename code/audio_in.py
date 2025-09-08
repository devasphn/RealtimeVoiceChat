import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    Manages audio input, processes it for transcription, and handles related callbacks.

    This class receives raw audio chunks, resamples them to the required format (16kHz),
    feeds them to an underlying `TranscriptionProcessor`, and manages callbacks for
    real-time transcription updates, recording start events, and silence detection.
    It also runs the transcription process in a background task.
    """

    _RESAMPLE_RATIO = 3  # Resample ratio from 48kHz (assumed input) to 16kHz.

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            pipeline_latency: float = 0.5,
        ) -> None:
        """
        Initializes the AudioInputProcessor.

        Args:
            language: Target language code for transcription (e.g., "en").
            is_orpheus: Flag indicating if a specific model variant should be used.
            silence_active_callback: Optional callback function invoked when silence state changes.
                                     It receives a boolean argument (True if silence is active).
            pipeline_latency: Estimated latency of the processing pipeline in seconds.
        """
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._silence_active_callback,
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        
        # Flag to indicate if the transcription loop has failed fatally
        self._transcription_failed = False
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())

        # Callbacks
        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[None], None]] = None
        self.silence_active_callback: Optional[Callable[[bool], None]] = silence_active_callback
        self.final_transcription_callback: Optional[Callable[[str], None]] = None  # NEW: Final transcription callback
        self.interrupted = False

        self._setup_callbacks()
        logger.info("👂🚀 AudioInputProcessor initialized.")

    def _silence_active_callback(self, is_active: bool) -> None:
        """Internal callback relay for silence detection status."""
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    def _on_recording_start(self) -> None:
        """Internal callback relay triggered when the transcriber starts recording."""
        if self.recording_start_callback:
            self.recording_start_callback()

    def abort_generation(self) -> None:
        """Signals the underlying transcriber to abort any ongoing generation process."""
        logger.info("👂🛑 Aborting generation requested.")
        self.transcriber.abort_generation()

    def _setup_callbacks(self) -> None:
        """Sets up internal callbacks for the TranscriptionProcessor instance."""
        def partial_transcript_callback(text: str) -> None:
            """Handles partial transcription results from the transcriber."""
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        def final_transcript_callback(text: str) -> None:
            """Handles final transcription results from the transcriber."""
            logger.info(f"👂✅ Final transcription received: '{text}'")
            if self.final_transcription_callback:
                try:
                    self.final_transcription_callback(text)
                except Exception as e:
                    logger.error(f"👂💥 Error in final transcription callback: {e}")

        self.transcriber.realtime_transcription_callback = partial_transcript_callback
        self.transcriber.set_final_transcription_callback(final_transcript_callback)

    async def _run_transcription_loop(self) -> None:
        """
        Continuously runs the transcription loop in a background asyncio task.

        It repeatedly calls the underlying `transcribe_loop`. If `transcribe_loop`
        finishes normally (completes one cycle), this loop calls it again.
        If `transcribe_loop` raises an Exception, it's treated as a fatal error,
        a flag is set, and this loop terminates. Handles CancelledError separately.
        """
        task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
        logger.info(f"👂▶️ Starting background transcription task ({task_name}).")
        while True:
            try:
                # Run one cycle of the underlying blocking loop
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                # If transcribe_loop returns without error, it means one cycle is complete.
                # The `while True` ensures it will be called again.
                logger.debug("👂✅ TranscriptionProcessor.transcribe_loop completed one cycle.")
                # Add a small sleep to prevent potential tight loop if transcribe_loop returns instantly
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 Transcription loop ({task_name}) cancelled.")
                # Do not set failure flag on cancellation
                break
            except Exception as e:
                # An actual error occurred within transcribe_loop
                logger.error(f"👂💥 Transcription loop ({task_name}) encountered a fatal error: {e}. Loop terminated.", exc_info=True)
                self._transcription_failed = True
                break

        logger.info(f"👂⏹️ Background transcription task ({task_name}) finished.")

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        Converts raw audio bytes (int16) to a 16kHz 16-bit PCM numpy array.

        The audio is converted to float32 for accurate resampling and then
        converted back to int16, clipping values outside the valid range.

        Args:
            raw_bytes: Raw audio data assumed to be in int16 format.

        Returns:
            A numpy array containing the resampled audio in int16 format at 16kHz.
            Returns an array of zeros if the input is silent.
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # Calculate expected length after resampling for silence
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        # Convert to float32 for resampling precision
        audio_float32 = raw_audio.astype(np.float32)

        # Resample using float32 data
        resampled_float = resample_poly(audio_float32, 1, self._RESAMPLE_RATIO)

        # Convert back to int16, clipping to ensure validity
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16

    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        Continuously processes audio chunks received from an asyncio Queue.

        Retrieves audio data, processes it using `process_audio_chunk`, and
        feeds the result to the transcriber unless interrupted or the transcription
        task has failed. Stops when `None` is received from the queue or upon error.

        Args:
            audio_queue: An asyncio queue expected to yield dictionaries containing
                         'pcm' (raw audio bytes) or None to terminate.
        """
        logger.info("👂▶️ Starting audio chunk processing loop.")
        while True:
            try:
                # Check if the transcription task has permanently failed *before* getting item
                if self._transcription_failed:
                    logger.error("👂🛑 Transcription task failed previously. Stopping audio processing.")
                    break

                # Check if the task finished unexpectedly (e.g., cancelled but not failed)
                if self.transcription_task and self.transcription_task.done() and not self._transcription_failed:
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        logger.error(f"👂🛑 Transcription task finished with unexpected error: {task_exception}. Stopping audio processing.", exc_info=task_exception)
                        self._transcription_failed = True
                        break
                    else:
                        logger.warning("👂⏹️ Transcription task is no longer running (completed or cancelled). Stopping audio processing.")
                        break

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal for audio processing.")
                    break

                pcm_data = audio_data.pop("pcm")

                # Process audio chunk (resampling happens consistently via float32)
                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    continue

                # Feed audio only if not interrupted and transcriber should be running
                if not self.interrupted:
                    if not self._transcription_failed:
                        # Feed audio to the underlying processor
                        self.transcriber.feed_audio(processed.tobytes(), audio_data)

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing task cancelled.")
                break
            except Exception as e:
                logger.error(f"👂💥 Audio processing error in queue loop: {e}", exc_info=True)

        logger.info("👂⏹️ Audio chunk processing loop finished.")

    def shutdown(self) -> None:
        """
        Initiates shutdown procedures for the audio processor and transcriber.

        Signals the transcriber to shut down and cancels the background
        transcription task.
        """
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        
        # Ensure transcriber shutdown is called first to signal the loop
        if hasattr(self.transcriber, 'shutdown'):
            logger.info("👂🛑 Signaling TranscriptionProcessor to shut down.")
            self.transcriber.shutdown()
        else:
            logger.warning("👂⚠️ TranscriptionProcessor does not have a shutdown method.")

        if self.transcription_task and not self.transcription_task.done():
            task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
            logger.info(f"👂🚫 Cancelling background transcription task ({task_name})...")
            self.transcription_task.cancel()

        logger.info("👂👋 AudioInputProcessor shutdown sequence initiated.")

    def set_final_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """Set the callback for final (complete) transcriptions"""
        self.final_transcription_callback = callback
        logger.info("👂🔗 Final transcription callback set")
