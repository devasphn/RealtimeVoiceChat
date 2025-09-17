# orpheus_server_manager.py
import os
import sys
import time
import subprocess
import threading
import logging
import requests
from typing import Optional
import signal
import atexit

logger = logging.getLogger(__name__)

class OrpheusServerManager:
    """
    Manages the llama-cpp-python server required for OrpheusEngine TTS.
    
    This class handles:
    - Starting the llama-cpp-python server with the Orpheus model
    - Checking if the server is already running
    - Ensuring the model file exists
    - Graceful shutdown of the server
    """
    
    def __init__(
        self,
        model_path: str = "/workspace/models/Orpheus-3b-FT-Q8_0.gguf",
        host: str = "0.0.0.0",
        port: int = 1234,
        n_gpu_layers: int = -1,
        timeout: int = 120
    ):
        """
        Initialize the Orpheus server manager.
        
        Args:
            model_path: Path to the Orpheus GGUF model file
            host: Host to bind the server to
            port: Port to run the server on
            n_gpu_layers: Number of GPU layers (-1 for all)
            timeout: Timeout in seconds to wait for server startup
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.timeout = timeout
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url = f"http://{host}:{port}"
        
        # Register cleanup on exit
        atexit.register(self.stop_server)
    
    def is_server_running(self) -> bool:
        """Check if the llama-cpp-python server is already running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.debug(f"🎤🔍 Server health check failed: {e}")
            return False

    def check_port_available(self) -> bool:
        """Check if the port is available for use."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    logger.warning(f"🎤⚠️ Port {self.port} is already in use")
                    return False
                return True
        except Exception as e:
            logger.error(f"🎤❌ Error checking port availability: {e}")
            return False

    def _get_model_size(self) -> str:
        """Get the size of the model file for logging."""
        try:
            if os.path.exists(self.model_path):
                size_bytes = os.path.getsize(self.model_path)
                size_gb = size_bytes / (1024**3)
                return f"{size_gb:.1f} GB"
            return "unknown"
        except Exception:
            return "unknown"
    
    def ensure_model_exists(self) -> bool:
        """Ensure the Orpheus model file exists."""
        if os.path.exists(self.model_path):
            logger.info(f"🎤✅ Orpheus model found at: {self.model_path}")
            return True
        
        # Try to create the models directory
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.error(f"🎤❌ Orpheus model not found at: {self.model_path}")
        logger.error("🎤💡 Please download the model using:")
        logger.error(f"   wget https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf -O {self.model_path}")
        return False
    
    def install_llama_cpp_python(self) -> bool:
        """Install llama-cpp-python with server support if not already installed."""
        try:
            import llama_cpp.server
            logger.info("🎤✅ llama-cpp-python[server] is already installed")
            return True
        except ImportError:
            logger.info("🎤⏬ Installing llama-cpp-python[server]...")
            try:
                # Install with CUDA support
                cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python[server]", 
                    "--force-reinstall", 
                    "--no-cache-dir"
                ]
                
                # Add CUDA compilation flags
                env = os.environ.copy()
                env["CMAKE_ARGS"] = "-DLLAMA_CUDA=on"
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info("🎤✅ llama-cpp-python[server] installed successfully")
                    return True
                else:
                    logger.error(f"🎤❌ Failed to install llama-cpp-python[server]: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error("🎤❌ Installation timed out")
                return False
            except Exception as e:
                logger.error(f"🎤❌ Installation failed: {e}")
                return False
    
    def start_server(self) -> bool:
        """Start the llama-cpp-python server for Orpheus."""
        if self.is_server_running():
            logger.info(f"🎤✅ Orpheus server already running at {self.server_url}")
            return True

        if not self.ensure_model_exists():
            return False

        if not self.install_llama_cpp_python():
            return False

        # Check if port is available
        if not self.check_port_available():
            logger.error(f"🎤❌ Port {self.port} is not available. Please check for conflicting processes.")
            logger.info(f"🎤💡 Try: lsof -ti:{self.port} | xargs kill -9")
            return False

        logger.info(f"🎤🚀 Starting Orpheus server at {self.server_url}...")
        logger.info(f"🎤📊 Model size: {self._get_model_size()}")
        logger.info(f"🎤⏱️ Timeout: {self.timeout} seconds")
        
        try:
            # Start the llama-cpp-python server
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--n_gpu_layers", str(self.n_gpu_layers)
            ]
            
            logger.info(f"🎤🔧 Running command: {' '.join(cmd)}")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            start_time = time.time()
            check_interval = 2
            last_log_time = start_time

            while time.time() - start_time < self.timeout:
                current_time = time.time()
                elapsed = current_time - start_time

                # Log progress every 10 seconds
                if current_time - last_log_time >= 10:
                    logger.info(f"🎤⏳ Waiting for Orpheus server startup... ({elapsed:.0f}s/{self.timeout}s)")
                    last_log_time = current_time

                if self.is_server_running():
                    logger.info(f"🎤✅ Orpheus server started successfully at {self.server_url} (took {elapsed:.1f}s)")
                    return True

                # Check if process died
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    logger.error(f"🎤❌ Server process died after {elapsed:.1f}s")
                    logger.error(f"🎤📤 stdout: {stdout}")
                    logger.error(f"🎤📤 stderr: {stderr}")
                    return False

                time.sleep(check_interval)

            # Timeout reached - get process output for debugging
            logger.error(f"🎤❌ Server failed to start within {self.timeout} seconds")
            if self.server_process and self.server_process.poll() is None:
                logger.info("🎤🔍 Server process still running, attempting to get output...")
                try:
                    # Give it a moment to produce output
                    stdout, stderr = self.server_process.communicate(timeout=5)
                    logger.error(f"🎤📤 Server stdout: {stdout}")
                    logger.error(f"🎤📤 Server stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning("🎤⏰ Could not get server output within timeout")

            self.stop_server()
            return False
            
        except Exception as e:
            logger.error(f"🎤❌ Failed to start Orpheus server: {e}")
            return False
    
    def stop_server(self):
        """Stop the llama-cpp-python server."""
        if self.server_process:
            logger.info("🎤🛑 Stopping Orpheus server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("🎤⚠️ Server didn't stop gracefully, forcing...")
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                logger.error(f"🎤❌ Error stopping server: {e}")
            finally:
                self.server_process = None
                logger.info("🎤✅ Orpheus server stopped")
    
    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError("Failed to start Orpheus server")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
