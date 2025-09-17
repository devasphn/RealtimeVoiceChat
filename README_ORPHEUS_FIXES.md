# Orpheus TTS Server Startup Fixes

## 🔍 Issues Identified and Fixed

### 1. **Shell Script Variable Substitution Error**
**Problem**: Line 196 in `install_complete_dependencies.sh` used `${torch.__version__}` which is Python syntax, not bash.

**Error**: `bad substitution` error in bash

**Fix**: 
```bash
# Before (broken):
echo -e "   • PyTorch ${torch.__version__} with CUDA support"

# After (fixed):
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo -e "   • PyTorch ${TORCH_VERSION} with CUDA support"
```

### 2. **Orpheus Server Startup Timeout**
**Problem**: Server was failing to start within 60-second timeout, especially for large 3GB model loading.

**Fix**: 
- Increased timeout from 60 to 120 seconds
- Added progress logging every 10 seconds
- Improved error handling and diagnostics

### 3. **Insufficient Error Diagnostics**
**Problem**: When server failed to start, there was minimal information about why.

**Fix**: Added comprehensive error logging:
- Port availability checking
- Model file size validation
- Detailed server output capture
- Process monitoring with timestamps

## 🔧 Files Modified

### 1. `install_complete_dependencies.sh`
- Fixed bash variable substitution error
- Now properly gets PyTorch version for summary

### 2. `code/orpheus_server_manager.py`
- Increased timeout from 60 to 120 seconds
- Added `check_port_available()` method
- Added `_get_model_size()` helper method
- Enhanced error logging with detailed output
- Added progress logging during startup
- Better process monitoring

### 3. `diagnose_orpheus_server.py` (New)
- Comprehensive diagnostic tool
- Tests all aspects of server startup
- Helps identify specific failure points

## 🚀 Updated Installation Process

### Step 1: Run Fixed Installation Script
```bash
cd /workspace/RealtimeVoiceChat
chmod +x install_complete_dependencies.sh
./install_complete_dependencies.sh
```

### Step 2: Diagnose Any Issues (Optional)
```bash
python diagnose_orpheus_server.py
```

### Step 3: Start Application
```bash
cd code
python server.py
```

## 🔍 Diagnostic Features Added

The new `OrpheusServerManager` now provides:

1. **Port Conflict Detection**:
   ```
   🎤⚠️ Port 1234 is already in use
   🎤💡 Try: lsof -ti:1234 | xargs kill -9
   ```

2. **Progress Monitoring**:
   ```
   🎤⏳ Waiting for Orpheus server startup... (10s/120s)
   🎤⏳ Waiting for Orpheus server startup... (20s/120s)
   ```

3. **Model Information**:
   ```
   🎤📊 Model size: 3.1 GB
   🎤⏱️ Timeout: 120 seconds
   ```

4. **Detailed Error Output**:
   ```
   🎤❌ Server process died after 15.2s
   🎤📤 stdout: [server output]
   🎤📤 stderr: [error details]
   ```

## 🧪 Troubleshooting Guide

### Common Issues and Solutions:

1. **Port 1234 already in use**:
   ```bash
   lsof -ti:1234 | xargs kill -9
   ```

2. **Model file missing or corrupted**:
   ```bash
   # Re-download model
   wget https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf \
        -O /workspace/models/Orpheus-3b-FT-Q8_0.gguf
   ```

3. **llama-cpp-python server module missing**:
   ```bash
   CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python[server] --force-reinstall
   ```

4. **CUDA not available**:
   - Check: `nvidia-smi`
   - Reinstall PyTorch with CUDA support

5. **Server startup timeout**:
   - Model loading can take 60-120 seconds for 3GB model
   - Check GPU memory availability
   - Monitor with diagnostic script

## 📊 Expected Startup Sequence

With the fixes, you should see:

```
🎤🚀 Initializing Orpheus server for TTS...
🎤✅ Orpheus model found at: /workspace/models/Orpheus-3b-FT-Q8_0.gguf
🎤✅ llama-cpp-python[server] is already installed
🎤🚀 Starting Orpheus server at http://0.0.0.0:1234...
🎤📊 Model size: 3.1 GB
🎤⏱️ Timeout: 120 seconds
🎤🔧 Running command: /usr/bin/python -m llama_cpp.server --model /workspace/models/Orpheus-3b-FT-Q8_0.gguf --host 0.0.0.0 --port 1234 --n_gpu_layers -1
🎤⏳ Waiting for Orpheus server startup... (10s/120s)
🎤⏳ Waiting for Orpheus server startup... (20s/120s)
🎤✅ Orpheus server started successfully at http://0.0.0.0:1234 (took 45.3s)
🎤✅ Orpheus server initialized successfully
🗣️🚀 SpeechPipelineManager initialized and workers started.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 🎯 Key Improvements

1. **Reliability**: Increased timeout and better error handling
2. **Diagnostics**: Comprehensive logging and diagnostic tools
3. **User Experience**: Clear progress indicators and helpful error messages
4. **Robustness**: Port conflict detection and model validation
5. **Debugging**: Detailed output capture for troubleshooting

The Orpheus TTS server should now start reliably without the previous timeout and diagnostic issues!
