# Orpheus TTS Configuration Fix

## Problem Summary

The RealtimeVoiceChat application was failing when using the Orpheus TTS engine because:

1. **OrpheusEngine requires a llama-cpp-python server** running on port 1234
2. **The application wasn't automatically starting this server**
3. **Missing llama-cpp-python[server] dependency**

## Root Cause

OrpheusEngine in RealtimeTTS is designed to connect to a llama-cpp-python server, not load the model directly. The configuration was:
- LLM: Ollama (port 11434) ✅
- TTS: Orpheus (needs llama-cpp-python server on port 1234) ❌

## Solution Overview

The fix includes:

1. **OrpheusServerManager**: Automatically manages the llama-cpp-python server
2. **Updated dependencies**: Added llama-cpp-python[server] to requirements.txt
3. **Integrated startup**: SpeechPipelineManager now starts Orpheus server automatically
4. **Setup scripts**: Automated installation and model download

## Files Modified/Added

### New Files:
- `code/orpheus_server_manager.py` - Manages Orpheus server lifecycle
- `setup_orpheus.sh` - Linux/RunPod setup script
- `setup_orpheus.bat` - Windows setup script
- `README_ORPHEUS_FIX.md` - This documentation

### Modified Files:
- `requirements.txt` - Added llama-cpp-python[server] and requests
- `code/speech_pipeline_manager.py` - Integrated OrpheusServerManager

## Installation Instructions

### For RunPod (Linux):

1. **Run the setup script:**
   ```bash
   cd /workspace/RealtimeVoiceChat
   chmod +x setup_orpheus.sh
   ./setup_orpheus.sh
   ```

2. **Install updated requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   cd code
   python server.py
   ```

### For Windows:

1. **Run the setup script:**
   ```cmd
   cd RealtimeVoiceChat
   setup_orpheus.bat
   ```

2. **Install updated requirements:**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```cmd
   cd code
   python server.py
   ```

### Manual Setup (if scripts fail):

1. **Create models directory:**
   ```bash
   mkdir -p /workspace/models  # Linux
   mkdir models                # Windows
   ```

2. **Download Orpheus model:**
   ```bash
   wget https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf -O /workspace/models/Orpheus-3b-FT-Q8_0.gguf
   ```

3. **Install llama-cpp-python with server support:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python[server] --force-reinstall --no-cache-dir
   ```

4. **Install other requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## How It Works

1. **Automatic Server Management**: When TTS engine is set to "orpheus", the SpeechPipelineManager automatically:
   - Checks if Orpheus model exists
   - Installs llama-cpp-python[server] if needed
   - Starts the llama-cpp-python server on port 1234
   - Waits for server to be ready before proceeding

2. **Health Checking**: The OrpheusServerManager continuously monitors server health

3. **Graceful Shutdown**: Server is properly stopped when the application shuts down

## Configuration

The Orpheus server runs with these default settings:
- **Host**: 0.0.0.0 (accessible from all interfaces)
- **Port**: 1234 (LM Studio compatible)
- **GPU Layers**: -1 (use all available GPU layers)
- **Model**: /workspace/models/Orpheus-3b-FT-Q8_0.gguf

## Troubleshooting

### Server Won't Start:
1. Check if model file exists: `/workspace/models/Orpheus-3b-FT-Q8_0.gguf`
2. Verify llama-cpp-python[server] is installed: `python -c "import llama_cpp.server"`
3. Check port 1234 isn't already in use: `netstat -an | grep 1234`

### Model Download Issues:
1. Ensure sufficient disk space (model is ~3GB)
2. Check internet connectivity
3. Try manual download with curl/wget

### CUDA Issues:
1. Verify CUDA is available: `nvidia-smi`
2. Reinstall with CUDA support: `CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python[server] --force-reinstall`

## Testing

To verify the fix works:

1. **Start the application** with Orpheus configuration
2. **Check logs** for "🎤✅ Orpheus server started successfully"
3. **Test TTS** by sending a message through the web interface
4. **Verify server** is running: `curl http://localhost:1234/health`

## Benefits

- ✅ **No manual server startup required**
- ✅ **Automatic model downloading**
- ✅ **Proper error handling and logging**
- ✅ **Graceful shutdown**
- ✅ **Health monitoring**
- ✅ **CUDA optimization**
