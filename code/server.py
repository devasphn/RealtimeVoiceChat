import asyncio
import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import local modules
from audio_in import AudioInputProcessor
from audio_module import AudioProcessor  
from llm_module import LLM
from logsetup import setup_logging

# === CONFIGURATION ===
START_ENGINE = "kokoro"  # Using kokoro instead of orpheus to avoid port conflicts
LLM_START_PROVIDER = "ollama"
LLM_START_MODEL = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M"
LANGUAGE = "en"
MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 150))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
USE_SSL = False
SSL_CERT_PATH = "localhost+3.pem"
SSL_KEY_PATH = "localhost+3-key.pem"

# Setup logging
setup_logging(getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Global instances
llm_instance: Optional[LLM] = None
audio_processor: Optional[AudioProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llm_instance, audio_processor
    
    logger.info("🚀 Starting Real-Time Voice Chat Server...")
    
    try:
        # Initialize LLM
        logger.info(f"🧠 Initializing LLM: {LLM_START_PROVIDER}/{LLM_START_MODEL}")
        system_prompt_path = Path("system_prompt.txt")
        system_prompt = ""
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text().strip()
            logger.info(f"📝 Loaded system prompt ({len(system_prompt)} chars)")
        else:
            logger.warning("📝 system_prompt.txt not found, using empty system prompt")
        
        llm_instance = LLM(
            backend=LLM_START_PROVIDER,
            model=LLM_START_MODEL,
            system_prompt=system_prompt
        )
        
        # Prewarm LLM
        logger.info("🔥 Prewarming LLM...")
        prewarm_success = llm_instance.prewarm(max_retries=1)
        if prewarm_success:
            logger.info("✅ LLM prewarmed successfully")
        else:
            logger.error("❌ LLM prewarm failed")
        
        # Initialize Audio Processor  
        logger.info(f"🎵 Initializing TTS Engine: {START_ENGINE}")
        audio_processor = AudioProcessor(engine=START_ENGINE)
        logger.info("✅ Audio processor initialized")
        
        logger.info("🎉 Server startup completed!")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("🧹 Server shutting down...")
        if llm_instance:
            llm_instance.cleanup_stale_requests(0)
        logger.info("👋 Server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Real-Time Voice Chat",
    description="Real-time voice conversation with AI using WebSockets",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning("📁 Static directory not found")

class VoiceChatSession:
    """Manages a single voice chat WebSocket session"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.is_active = False
        self.audio_input: Optional[AudioInputProcessor] = None
        self.conversation_history = []
        
        # Audio queues
        self.incoming_audio_queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE_SIZE)
        self.outgoing_audio_queue = Queue(maxsize=MAX_AUDIO_QUEUE_SIZE * 2)
        
        # Control flags
        self.is_generating = False
        self.should_stop = threading.Event()
        
        # Background tasks
        self.audio_sender_task: Optional[asyncio.Task] = None
        self.audio_processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "audio_chunks_received": 0,
            "audio_chunks_sent": 0, 
            "messages_processed": 0,
            "responses_generated": 0,
            "transcriptions_completed": 0
        }
        
    async def start(self):
        """Start the voice chat session"""
        if self.is_active:
            return
            
        logger.info("🎤 Starting voice chat session")
        self.is_active = True
        
        try:
            # Initialize audio input processor
            self.audio_input = AudioInputProcessor(
                language=LANGUAGE,
                is_orpheus=(START_ENGINE == "orpheus"),
                silence_active_callback=self._on_silence_change,
                pipeline_latency=0.5
            )
            
            # Setup audio input callbacks
            self.audio_input.realtime_callback = self._on_partial_transcription
            self.audio_input.recording_start_callback = self._on_recording_start
            
            # CRITICAL: Set the final transcription callback to process complete speech
            self.audio_input.set_final_transcription_callback(self._process_complete_speech)
            
            # Setup TTS callback
            if audio_processor:
                audio_processor.on_first_audio_chunk_synthesize = self._on_first_tts_chunk
            
            # Start background tasks
            self.audio_sender_task = asyncio.create_task(self._audio_sender_loop())
            self.audio_processor_task = asyncio.create_task(self.audio_input.process_chunk_queue(self.incoming_audio_queue))
            
            await self._send_status("session_started")
            logger.info("✅ Voice chat session started successfully")
            
        except Exception as e:
            logger.error(f"💥 Failed to start session: {e}", exc_info=True)
            await self._send_error(f"Session start failed: {e}")
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up session resources"""
        logger.info("🧹 Cleaning up voice chat session")
        self.is_active = False
        self.should_stop.set()
        
        # Cancel background tasks
        if self.audio_sender_task and not self.audio_sender_task.done():
            self.audio_sender_task.cancel()
            try:
                await self.audio_sender_task
            except asyncio.CancelledError:
                pass
        
        if self.audio_processor_task and not self.audio_processor_task.done():
            self.audio_processor_task.cancel()
            try:
                await self.audio_processor_task
            except asyncio.CancelledError:
                pass
        
        # Send termination signal to audio processing
        try:
            self.incoming_audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        
        # Cleanup audio input
        if self.audio_input:
            self.audio_input.shutdown()
        
        logger.info("✅ Session cleanup completed")
    
    async def handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            self.stats["messages_processed"] += 1
            
            if message_type == "audio_chunk":
                await self._handle_audio_chunk(data)
            elif message_type == "start_recording":
                await self._handle_start_recording()
            elif message_type == "stop_recording":
                await self._handle_stop_recording()
            elif message_type == "reset_conversation":
                await self._handle_reset_conversation()
            elif message_type == "interrupt":
                await self._handle_interrupt()
            elif message_type == "get_stats":
                await self._send_stats()
            else:
                logger.warning(f"❓ Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"📄 JSON decode error: {e}")
            await self._send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"💥 Message handling error: {e}", exc_info=True)
            await self._send_error(f"Message processing failed: {e}")
    
    async def _handle_audio_chunk(self, data: Dict[str, Any]):
        """Process incoming audio chunk"""
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data["audio_data"])
            
            # Add to processing queue
            chunk_info = {
                "pcm": audio_data,
                "timestamp": time.time(),
                "chunk_id": self.stats["audio_chunks_received"]
            }
            
            try:
                self.incoming_audio_queue.put_nowait(chunk_info)
                self.stats["audio_chunks_received"] += 1
            except asyncio.QueueFull:
                logger.warning("📥 Incoming audio queue full, dropping chunk")
                
        except Exception as e:
            logger.error(f"🎤 Audio chunk processing error: {e}")
    
    async def _handle_start_recording(self):
        """Handle recording start"""
        logger.info("🎙️ Recording started")
        await self._send_status("recording_started")
    
    async def _handle_stop_recording(self):
        """Handle recording stop"""  
        logger.info("🎙️ Recording stopped")
        await self._send_status("recording_stopped")
    
    async def _handle_reset_conversation(self):
        """Reset conversation history"""
        logger.info("🔄 Resetting conversation")
        self.conversation_history.clear()
        await self._send_status("conversation_reset")
    
    async def _handle_interrupt(self):
        """Handle user interruption"""
        logger.info("🛑 Processing interruption")
        
        self.is_generating = False
        self.should_stop.set()
        
        # Clear outgoing audio queue
        try:
            while not self.outgoing_audio_queue.empty():
                self.outgoing_audio_queue.get_nowait()
        except Empty:
            pass
        
        # Abort audio processing
        if self.audio_input:
            self.audio_input.abort_generation()
        
        # Reset stop event for next interaction
        self.should_stop.clear()
        
        await self._send_status("interrupted")
    
    async def _audio_sender_loop(self):
        """Send audio chunks to client via WebSocket"""
        logger.info("🎵 Starting audio sender loop")
        
        try:
            while self.is_active:
                try:
                    # Get audio chunk from queue with timeout
                    try:
                        audio_chunk = self.outgoing_audio_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    
                    if audio_chunk is None:  # Shutdown signal
                        break
                    
                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    
                    # Send to client
                    message = {
                        "type": "audio_chunk",
                        "audio_data": audio_b64,
                        "timestamp": time.time(),
                        "chunk_id": self.stats["audio_chunks_sent"]
                    }
                    
                    await self.websocket.send_text(json.dumps(message))
                    self.stats["audio_chunks_sent"] += 1
                    
                    # Small delay to prevent overwhelming client
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"🎵 Audio sender error: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("🎵 Audio sender cancelled")
        except Exception as e:
            logger.error(f"🎵 Audio sender error: {e}", exc_info=True)
        finally:
            logger.info("🎵 Audio sender loop stopped")
    
    def _on_partial_transcription(self, text: str):
        """Handle partial transcription from STT"""
        try:
            asyncio.create_task(self._send_partial_transcription(text))
        except Exception as e:
            logger.error(f"📝 Partial transcription callback error: {e}")
    
    def _on_recording_start(self):
        """Handle recording start detected by STT"""
        try:
            asyncio.create_task(self._send_status("recording_detected"))
        except Exception as e:
            logger.error(f"🎙️ Recording start callback error: {e}")
    
    def _on_silence_change(self, is_silent: bool):
        """Handle silence state change from STT"""
        try:
            status = "silence_detected" if is_silent else "voice_detected"
            asyncio.create_task(self._send_status(status))
        except Exception as e:
            logger.error(f"🔇 Silence callback error: {e}")
    
    def _on_first_tts_chunk(self):
        """Handle first TTS audio chunk generated"""
        logger.info("🎵 First TTS chunk generated")
        try:
            asyncio.create_task(self._send_status("tts_started"))
        except Exception as e:
            logger.error(f"🎵 TTS callback error: {e}")
    
    def _process_complete_speech(self, user_text: str):
        """Process complete user speech and generate AI response"""
        if not user_text or not user_text.strip():
            logger.warning("📝 Empty speech text received")
            return
            
        logger.info(f"🗣️ Complete speech received: '{user_text[:100]}{'...' if len(user_text) > 100 else ''}'")
        
        # Update statistics
        self.stats["transcriptions_completed"] += 1
        
        # Send final transcription to client
        asyncio.create_task(self._send_final_transcription(user_text))
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": user_text
        })
        
        # Generate response in background thread
        def response_thread():
            try:
                self.is_generating = True
                logger.info("🧠 Starting LLM generation...")
                
                # Send status update
                asyncio.create_task(self._send_status("generating_response"))
                
                # Generate LLM response
                response_text = ""
                if llm_instance:
                    try:
                        generation = llm_instance.generate(
                            text=user_text,
                            history=self.conversation_history[:-1]  # Exclude current user message
                        )
                        
                        # Collect tokens
                        for token in generation:
                            if not self.is_active or self.should_stop.is_set():
                                logger.info("🧠 LLM generation interrupted")
                                break
                            response_text += token
                        
                        if response_text.strip() and self.is_active and not self.should_stop.is_set():
                            # Add to conversation history
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            
                            self.stats["responses_generated"] += 1
                            
                            # Send response to client
                            asyncio.create_task(self._send_final_response(response_text))
                            
                            # Start TTS synthesis
                            logger.info("🎵 Starting TTS synthesis...")
                            self._start_tts_synthesis(response_text)
                        else:
                            logger.warning("🧠 Empty or interrupted LLM response")
                            
                    except Exception as e:
                        logger.error(f"🧠 LLM generation error: {e}", exc_info=True)
                        asyncio.create_task(self._send_error(f"LLM generation failed: {e}"))
                else:
                    logger.error("🧠 No LLM instance available")
                    asyncio.create_task(self._send_error("LLM not available"))
                    
            except Exception as e:
                logger.error(f"🧠 Response generation error: {e}", exc_info=True)
                asyncio.create_task(self._send_error(f"Response generation failed: {e}"))
            finally:
                self.is_generating = False
                asyncio.create_task(self._send_status("generation_complete"))
        
        # Start in background thread
        threading.Thread(target=response_thread, daemon=True).start()
    
    def _start_tts_synthesis(self, text: str):
        """Start TTS synthesis in background thread"""
        if not audio_processor:
            logger.error("🎵 No audio processor available")
            return
            
        def synthesis_thread():
            try:
                logger.info(f"🎵 TTS synthesis starting for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Create stop event that respects the session's stop signal
                stop_event = threading.Event()
                
                # Monitor session stop signal in background
                def monitor_stop():
                    while not stop_event.is_set():
                        if self.should_stop.is_set() or not self.is_active:
                            stop_event.set()
                            break
                        time.sleep(0.1)
                
                monitor_thread = threading.Thread(target=monitor_stop, daemon=True)
                monitor_thread.start()
                
                # Synthesize audio
                success = audio_processor.synthesize(
                    text=text,
                    audio_chunks=self.outgoing_audio_queue,
                    stop_event=stop_event,
                    generation_string="RESPONSE"
                )
                
                if success:
                    logger.info("🎵 TTS synthesis completed successfully")
                    asyncio.create_task(self._send_status("tts_complete"))
                else:
                    logger.warning("🎵 TTS synthesis failed or was interrupted")
                    
                # Signal end of audio stream
                try:
                    self.outgoing_audio_queue.put_nowait(None)
                except Exception as e:
                    logger.warning(f"🎵 Could not signal end of audio stream: {e}")
                    
            except Exception as e:
                logger.error(f"🎵 TTS synthesis error: {e}", exc_info=True)
                asyncio.create_task(self._send_error(f"TTS synthesis failed: {e}"))
        
        threading.Thread(target=synthesis_thread, daemon=True).start()
    
    # WebSocket message sending methods
    async def _send_status(self, status: str):
        """Send status message to client"""
        try:
            message = {
                "type": "status",
                "status": status,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📤 Status send error: {e}")
    
    async def _send_partial_transcription(self, text: str):
        """Send partial transcription to client"""
        try:
            message = {
                "type": "partial_transcription",
                "text": text,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📤 Partial transcription send error: {e}")
    
    async def _send_final_transcription(self, text: str):
        """Send final transcription to client"""
        try:
            message = {
                "type": "final_transcription",
                "text": text,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📤 Final transcription send error: {e}")
    
    async def _send_final_response(self, text: str):
        """Send final AI response to client"""
        try:
            message = {
                "type": "final_response",
                "text": text,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📤 Final response send error: {e}")
    
    async def _send_error(self, error: str):
        """Send error message to client"""
        try:
            message = {
                "type": "error",
                "error": error,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📤 Error send error: {e}")
    
    async def _send_stats(self):
        """Send session statistics to client"""
        try:
            stats = {
                **self.stats,
                "is_active": self.is_active,
                "is_generating": self.is_generating,
                "conversation_length": len(self.conversation_history),
                "incoming_queue_size": self.incoming_audio_queue.qsize(),
                "outgoing_queue_size": self.outgoing_audio_queue.qsize()
            }
            
            message = {
                "type": "stats",
                "stats": stats,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"📊 Stats send error: {e}")

# FastAPI Routes
@app.get("/")
async def get_index():
    """Serve the main HTML page"""
    index_file = Path("static/index.html")
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Voice Chat</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                .error { color: red; background: #ffe6e6; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Real-Time Voice Chat</h1>
            <div class="error">
                <h2>Static Files Not Found</h2>
                <p>The 'static' directory with index.html was not found. Please ensure you have:</p>
                <ul>
                    <li>A 'static' directory in the same location as server.py</li>
                    <li>An 'index.html' file inside the static directory</li>
                    <li>Other required static files (app.js, etc.)</li>
                </ul>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "llm_available": llm_instance is not None,
        "audio_processor_available": audio_processor is not None,
        "engine": START_ENGINE,
        "llm_provider": LLM_START_PROVIDER,
        "llm_model": LLM_START_MODEL
    }

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice chat"""
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info(f"🔌 WebSocket connected: {client_info}")
    
    # Create session
    session = VoiceChatSession(websocket)
    
    try:
        # Start session
        await session.start()
        
        # Main message loop
        while session.is_active:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=1.0
                )
                await session.handle_message(message)
                
            except asyncio.TimeoutError:
                # Continue loop - this is normal
                continue
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected by client: {client_info}")
                break
            except Exception as e:
                logger.error(f"💥 WebSocket error: {e}", exc_info=True)
                break
        
    except Exception as e:
        logger.error(f"💥 WebSocket session error: {e}", exc_info=True)
    finally:
        await session.cleanup()
        logger.info(f"🔌 WebSocket session ended: {client_info}")

if __name__ == "__main__":
    # Configuration
    host = "0.0.0.0"
    port = 8000
    
    if USE_SSL:
        cert_path = Path(SSL_CERT_PATH)
        key_path = Path(SSL_KEY_PATH)
        
        if not (cert_path.exists() and key_path.exists()):
            logger.error(f"❌ SSL certificates not found!")
            logger.error(f"   Certificate: {cert_path.absolute()}")
            logger.error(f"   Key: {key_path.absolute()}")
            exit(1)
        
        logger.info(f"🔐 Starting HTTPS server on https://{host}:{port}")
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            log_level=LOG_LEVEL.lower(),
            access_log=True
        )
    else:
        logger.info(f"🌐 Starting HTTP server on http://{host}:{port}")
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            log_level=LOG_LEVEL.lower(),
            access_log=True,
            reload=False
        )
