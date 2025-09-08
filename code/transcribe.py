import logging
import threading
from typing import Callable, Optional

from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

# Default configuration for the recorder
DEFAULT_RECORDER_CONFIG = {
    "model": "base.en",
    "language": "en",
    "silence_limit_seconds": 2.0,
    "noise_reduction_enabled": True,
    "early_transcription_on_silence": 1.0,
    "beam_size": 5,
    "initial_prompt": None,
    "use_microphone": False,  # We'll feed audio manually
    "spinner": False,
    "level": logging.WARNING,  # Reduce verbosity
}

class TranscriptionProcessor:
    """
    Handles real-time speech-to-text transcription using RealtimeSTT.
    
    This class processes audio chunks fed to it and provides callbacks for
    partial transcriptions, recording start events, and final transcriptions.
    It runs the transcription process in a separate thread to avoid blocking.
    """
    
    def __init__(
        self,
        language: str = "en",
        on_recording_start_callback: Optional[Callable[[], None]] = None,
        silence_active_callback: Optional[Callable[[bool], None]] = None,
        is_orpheus: bool = False,
        pipeline_latency: float = 0.5,
        recorder_config: Optional[dict] = None,
    ):
        """
        Initialize the transcription processor.
        
        Args:
            language: Target language for transcription (e.g., "en")
            on_recording_start_callback: Called when recording/transcription starts
            silence_active_callback: Called when silence state changes
            is_orpheus: Special flag for Orpheus engine compatibility
            pipeline_latency: Expected pipeline latency in seconds
            recorder_config: Custom recorder configuration (overrides defaults)
        """
        self.language = language
        self.on_recording_start_callback = on_recording_start_callback
        self.silence_active_callback = silence_active_callback
        self.is_orpheus = is_orpheus
        self.pipeline_latency = pipeline_latency
        
        # Callback for real-time (partial) transcriptions
        self.realtime_transcription_callback: Optional[Callable[[str], None]] = None
        
        # Callback for complete transcriptions (full sentences)
        self.final_transcription_callback: Optional[Callable[[str], None]] = None
        
        # Internal state
        self._recorder: Optional[AudioToTextRecorder] = None
        self._transcription_thread: Optional[threading.Thread] = None
        self._shutdown_requested = False
        self._is_recording = False
        
        # Prepare recorder configuration
        config = DEFAULT_RECORDER_CONFIG.copy()
        if recorder_config:
            config.update(recorder_config)
        
        # Apply language-specific settings
        config["language"] = language
        config["model"] = f"base.{language}" if language != "en" else "base.en"
        
        # Adjust silence settings for real-time use
        config["silence_limit_seconds"] = 1.5  # Shorter silence limit for responsiveness
        config["early_transcription_on_silence"] = 0.8
        
        self._config = config
        logger.info(f"👂 TranscriptionProcessor initialized for language '{language}'")
    
    def _initialize_recorder(self):
        """Initialize the AudioToTextRecorder with callbacks"""
        try:
            self._recorder = AudioToTextRecorder(
                model=self._config["model"],
                language=self._config["language"],
                silence_limit_seconds=self._config["silence_limit_seconds"],
                noise_reduction_enabled=self._config["noise_reduction_enabled"],
                early_transcription_on_silence=self._config["early_transcription_on_silence"],
                beam_size=self._config["beam_size"],
                initial_prompt=self._config["initial_prompt"],
                use_microphone=self._config["use_microphone"],
                spinner=self._config["spinner"],
                level=self._config["level"],
                
                # Callbacks
                on_recording_start=self._on_recording_start_internal,
                on_transcription_start=self._on_transcription_start_internal,
                on_realtime_transcription_update=self._on_realtime_transcription_internal,
                on_realtime_transcription_stabilized=self._on_stabilized_transcription_internal,
            )
            logger.info("👂 AudioToTextRecorder initialized successfully")
            
        except Exception as e:
            logger.error(f"👂 Failed to initialize recorder: {e}", exc_info=True)
            raise
    
    def _on_recording_start_internal(self):
        """Internal callback when recording starts"""
        logger.info("👂 Recording start detected")
        self._is_recording = True
        
        # Call silence callback to indicate voice activity
        if self.silence_active_callback:
            try:
                self.silence_active_callback(False)  # False = not silent, voice detected
            except Exception as e:
                logger.error(f"👂 Error in silence callback: {e}")
        
        # Call external recording start callback
        if self.on_recording_start_callback:
            try:
                self.on_recording_start_callback()
            except Exception as e:
                logger.error(f"👂 Error in recording start callback: {e}")
    
    def _on_transcription_start_internal(self):
        """Internal callback when transcription processing starts"""
        logger.debug("👂 Transcription processing started")
    
    def _on_realtime_transcription_internal(self, text: str):
        """Internal callback for partial/realtime transcriptions"""
        if not text or not text.strip():
            return
            
        logger.debug(f"👂 Partial transcription: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        if self.realtime_transcription_callback:
            try:
                self.realtime_transcription_callback(text)
            except Exception as e:
                logger.error(f"👂 Error in realtime transcription callback: {e}")
    
    def _on_stabilized_transcription_internal(self, text: str):
        """Internal callback for stabilized/final transcriptions"""
        if not text or not text.strip():
            return
        
        logger.info(f"👂 Final transcription: '{text}'")
        
        # Call silence callback to indicate silence (end of speech)
        if self.silence_active_callback:
            try:
                self.silence_active_callback(True)  # True = silence detected
            except Exception as e:
                logger.error(f"👂 Error in silence callback: {e}")
        
        # Trigger final transcription callback (this should start LLM + TTS)
        if self.final_transcription_callback:
            try:
                logger.info(f"👂 Triggering final transcription callback with: '{text}'")
                self.final_transcription_callback(text)
            except Exception as e:
                logger.error(f"👂 Error in final transcription callback: {e}")
        else:
            logger.warning("👂 No final transcription callback set!")
        
        self._is_recording = False
    
    def transcribe_loop(self):
        """Main transcription loop (blocking, should run in separate thread)"""
        if not self._recorder:
            self._initialize_recorder()
            
        if not self._recorder:
            logger.error("👂 Failed to initialize recorder")
            return
        
        logger.info("👂 Starting transcription loop...")
        
        try:
            while not self._shutdown_requested:
                try:
                    # This call blocks until transcription is complete
                    final_text = self._recorder.text()
                    
                    if final_text and final_text.strip() and not self._shutdown_requested:
                        logger.info(f"👂 Transcription loop got final text: '{final_text}'")
                        
                        # The stabilized callback should have already been called,
                        # but call it again as a safety measure
                        self._on_stabilized_transcription_internal(final_text)
                    
                except Exception as e:
                    logger.error(f"👂 Error in transcription loop: {e}", exc_info=True)
                    # Wait a bit before retrying
                    import time
                    time.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"👂 Fatal error in transcription loop: {e}", exc_info=True)
        finally:
            logger.info("👂 Transcription loop ended")
    
    def feed_audio(self, audio_data: bytes, metadata: dict = None):
        """
        Feed audio data to the transcription system.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM, 16kHz)
            metadata: Additional metadata (currently unused)
        """
        if not self._recorder or self._shutdown_requested:
            return
            
        try:
            # Convert bytes to numpy array and feed to recorder
            import numpy as np
            
            # Assuming audio_data is 16-bit PCM
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Feed to the recorder
            self._recorder.feed_audio(audio_np)
            
        except Exception as e:
            logger.error(f"👂 Error feeding audio: {e}")
    
    def abort_generation(self):
        """Signal the transcription to abort/reset"""
        logger.info("👂 Aborting transcription generation")
        
        if self._recorder:
            try:
                # Try to interrupt the current transcription
                self._recorder.abort()
            except Exception as e:
                logger.warning(f"👂 Error aborting recorder: {e}")
        
        self._is_recording = False
    
    def shutdown(self):
        """Shutdown the transcription processor"""
        logger.info("👂 Shutting down transcription processor...")
        
        self._shutdown_requested = True
        
        # Abort current transcription
        self.abort_generation()
        
        # Close recorder
        if self._recorder:
            try:
                self._recorder.shutdown()
            except Exception as e:
                logger.warning(f"👂 Error shutting down recorder: {e}")
            finally:
                self._recorder = None
        
        logger.info("👂 Transcription processor shutdown complete")
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording/transcribing"""
        return self._is_recording and not self._shutdown_requested
    
    def set_final_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback for final (complete) transcriptions"""
        self.final_transcription_callback = callback
        logger.info("👂 Final transcription callback set")
    
    def get_stats(self) -> dict:
        """Get transcription statistics"""
        return {
            "is_recording": self.is_recording,
            "shutdown_requested": self._shutdown_requested,
            "recorder_initialized": self._recorder is not None,
            "language": self.language,
            "config": self._config
        }
