# Complete RealtimeVoiceChat Setup Guide

## 🎯 Overview

This guide provides a comprehensive solution for setting up RealtimeVoiceChat on RunPod with all dependencies properly installed, including the fix for the `scipy` import error and complete Orpheus TTS configuration.

## 🔍 Root Cause Analysis

The application was failing due to multiple missing dependencies:

1. **Missing `scipy`** - Required by `upsample_overlap.py` and `audio_in.py` for audio resampling
2. **Missing `numpy`** - Core dependency for audio processing
3. **Missing `transformers`** - Required by `turndetect.py` for sentence classification
4. **Missing `huggingface_hub`** - Required by `audio_module.py` for model downloads
5. **Missing system audio libraries** - Required for audio processing
6. **Orpheus server not auto-starting** - Fixed in previous solution

## 📋 Complete Dependency List

### Core ML & Scientific Computing:
- `torch==2.5.1+cu121` (with CUDA support)
- `torchaudio==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `numpy` (audio processing, arrays)
- `scipy` (signal processing, resampling)
- `transformers` (turn detection, NLP models)

### Speech Processing:
- `realtimestt==0.3.104` (speech-to-text)
- `realtimetts[kokoro,coqui,orpheus]==0.5.7` (text-to-speech)
- `llama-cpp-python[server]` (Orpheus TTS backend)

### Audio & Model Management:
- `huggingface_hub` (model downloads)

### Web Server:
- `fastapi` (web framework)
- `uvicorn` (ASGI server)

### Utilities:
- `python-dotenv` (configuration)
- `requests` (HTTP requests)
- `openai` (OpenAI API client)
- `ollama` (Ollama client)

### System Dependencies:
- `libsndfile1-dev` (audio file handling)
- `portaudio19-dev` (audio I/O)
- `libasound2-dev` (ALSA audio)
- `libpulse-dev` (PulseAudio)
- `ffmpeg` (audio/video processing)

## 🚀 Installation Instructions

### Option 1: Automated Installation (Recommended)

Run the complete installation script:

```bash
cd /workspace/RealtimeVoiceChat
chmod +x install_complete_dependencies.sh
./install_complete_dependencies.sh
```

### Option 2: Manual Step-by-Step Installation

1. **Update system and install audio dependencies:**
   ```bash
   apt-get update
   apt-get install -y build-essential python3-dev libsndfile1-dev \
       portaudio19-dev libasound2-dev libpulse-dev ffmpeg git curl wget
   ```

2. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 \
       --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install llama-cpp-python with CUDA and server support:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python[server] --force-reinstall
   ```

4. **Install scientific computing libraries:**
   ```bash
   pip install numpy scipy transformers huggingface_hub
   ```

5. **Install speech processing libraries:**
   ```bash
   pip install realtimestt==0.3.104 "realtimetts[kokoro,coqui,orpheus]==0.5.7"
   ```

6. **Install web server and utilities:**
   ```bash
   pip install fastapi uvicorn python-dotenv requests openai ollama
   ```

7. **Install compatibility constraints:**
   ```bash
   pip install "ctranslate2<4.5.0"
   ```

8. **Download Orpheus model:**
   ```bash
   mkdir -p /workspace/models
   wget https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf \
       -O /workspace/models/Orpheus-3b-FT-Q8_0.gguf
   ```

## 🧪 Verification

Run the verification script to ensure all dependencies are properly installed:

```bash
python verify_dependencies.py
```

This will check:
- ✅ All Python package imports
- ✅ PyTorch CUDA configuration
- ✅ Orpheus model file existence and size
- ✅ System audio dependencies

## 🎮 Starting the Application

After successful installation:

```bash
cd /workspace/RealtimeVoiceChat/code
python server.py
```

You should see logs like:
```
🎤🚀 Initializing Orpheus server for TTS...
🎤✅ Orpheus model found at: /workspace/models/Orpheus-3b-FT-Q8_0.gguf
🎤✅ llama-cpp-python[server] is already installed
🎤🚀 Starting Orpheus server at http://0.0.0.0:1234...
🎤✅ Orpheus server started successfully
🗣️🚀 SpeechPipelineManager initialized and workers started.
```

## 🔧 Troubleshooting

### Common Issues and Solutions:

1. **`ModuleNotFoundError: No module named 'scipy'`**
   - **Solution**: Run `pip install scipy`

2. **`ModuleNotFoundError: No module named 'numpy'`**
   - **Solution**: Run `pip install numpy`

3. **`ModuleNotFoundError: No module named 'transformers'`**
   - **Solution**: Run `pip install transformers`

4. **CUDA not available**
   - **Solution**: Ensure NVIDIA drivers are installed and PyTorch was installed with CUDA support

5. **Orpheus model not found**
   - **Solution**: Download manually or run the installation script

6. **Audio processing errors**
   - **Solution**: Install system audio dependencies with apt-get

7. **Port 1234 already in use**
   - **Solution**: Kill existing processes on port 1234: `lsof -ti:1234 | xargs kill -9`

### Verification Commands:

```bash
# Test core imports
python -c "import numpy, scipy, torch, transformers; print('✅ Core libraries OK')"

# Test speech libraries
python -c "from RealtimeSTT import AudioToTextRecorder; from RealtimeTTS import OrpheusEngine; print('✅ Speech libraries OK')"

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test llama-cpp-python server
python -c "import llama_cpp.server; print('✅ llama-cpp-python server OK')"
```

## 📁 File Structure

After installation, your structure should look like:

```
RealtimeVoiceChat/
├── code/                           # Application code
├── models/                         # Model files
│   └── Orpheus-3b-FT-Q8_0.gguf    # Orpheus TTS model (~3GB)
├── requirements.txt                # Updated with all dependencies
├── install_complete_dependencies.sh # Complete installation script
├── verify_dependencies.py         # Dependency verification
└── README_COMPLETE_SETUP.md       # This guide
```

## 🎉 Success Indicators

When everything is working correctly:

1. ✅ All imports succeed without errors
2. ✅ PyTorch reports CUDA is available
3. ✅ Orpheus server starts automatically on port 1234
4. ✅ Web server starts on port 8000
5. ✅ Audio processing works without scipy errors
6. ✅ TTS synthesis produces audio output

## 🔄 Configuration

The application is configured for:
- **ASR**: FasterWhisper (via RealtimeSTT)
- **LLM**: Mistral via Ollama (port 11434)
- **TTS**: Orpheus via llama-cpp-python server (port 1234)

This configuration now works seamlessly without manual intervention!
