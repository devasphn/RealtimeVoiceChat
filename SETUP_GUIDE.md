# RealtimeVoiceChat Setup Guide

## 🎯 Complete Setup for Robust Operation

This guide provides a comprehensive setup for the RealtimeVoiceChat application with automatic dependency management and graceful error handling.

## 🚀 Quick Start (Recommended)

### 1. Complete Automated Setup
```bash
# Run the complete setup script (handles everything)
chmod +x setup_complete.sh
./setup_complete.sh
```

### 2. Start Application
```bash
# Use the robust startup script
chmod +x start_app.sh
./start_app.sh
```

### 3. Access Application
- **Main Application**: http://localhost:8000
- **Ollama API**: http://localhost:11434 (LLM backend)
- **TTS Server**: http://localhost:1234 (Orpheus TTS - optional)

## 🔧 What the Setup Includes

### Automatic Installation & Configuration:
- ✅ **System Dependencies**: Audio libraries, build tools, Python dev packages
- ✅ **Ollama**: Automatic installation and service startup
- ✅ **Mistral 7B Model**: Downloaded and ready for LLM processing
- ✅ **Whisper Model**: Base model for speech recognition
- ✅ **Python Packages**: All required dependencies from requirements.txt
- ✅ **Audio System**: ALSA/PulseAudio configuration to suppress warnings
- ✅ **TTS Server**: Optional Orpheus server (graceful fallback if not available)

### Robust Error Handling:
- 🛡️ **Graceful Fallbacks**: Application continues even if TTS server fails
- 🛡️ **Dependency Checks**: Verifies all components before startup
- 🛡️ **Clear Logging**: Detailed status information and helpful error messages
- 🛡️ **Manual Override**: Instructions for manual TTS server startup if needed

## 📋 Manual Setup (Alternative)

If you prefer step-by-step manual setup:

### Step 1: System Dependencies
```bash
# Update system
apt-get update

# Install essential packages
apt-get install -y curl wget git build-essential cmake python3-dev python3-pip

# Install audio libraries
apt-get install -y libsndfile1-dev portaudio19-dev libasound2-dev libpulse-dev alsa-utils ffmpeg
```

### Step 2: Python Dependencies
```bash
# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Wait for service to start, then download model
sleep 10
ollama pull mistral:7b
```

### Step 4: Audio Configuration
```bash
# Set audio environment variables
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime
export SDL_AUDIODRIVER=pulse
```

### Step 5: Start Application
```bash
cd code
python server.py
```

## 🔍 Troubleshooting

### Common Issues and Solutions:

#### 1. Ollama Connection Refused
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve &

# Check if model is available
ollama list
```

#### 2. TTS Server Not Starting
The application now handles this gracefully. If you want TTS functionality:
```bash
# Start TTS server manually
python -m llama_cpp.server \
  --model /workspace/models/Orpheus-3b-FT-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 1234 \
  --n_gpu_layers -1
```

#### 3. Audio Warnings/Errors
```bash
# Source the audio environment
source set_audio_env.sh

# Or set manually
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
```

#### 4. Missing Python Packages
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### 5. Whisper Model Issues
```bash
# Download Whisper model manually
python -c "import whisper; whisper.load_model('base', download_root='/workspace/models')"
```

## 📊 Application Status Checks

### Check Ollama Status
```bash
# Run the status check script (created during setup)
./check_ollama_status.sh
```

### Check All Services
```bash
# Ollama API
curl http://localhost:11434/api/tags

# TTS Server (optional)
curl http://localhost:1234/health

# Main Application
curl http://localhost:8000
```

## 🎛️ Configuration

### Model Configuration
Edit `code/server.py` to change models:
```python
# LLM Model (Ollama)
LLM_START_MODEL = "mistral:7b"  # Change to other Ollama models

# TTS Engine
TTS_START_ENGINE = "orpheus"    # Options: orpheus, kokoro, coqui

# Whisper Model (in setup scripts)
WHISPER_MODEL = "base"          # Options: tiny, base, small, medium, large
```

### Audio Configuration
Edit `set_audio_env.sh` for custom audio settings:
```bash
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime
export SDL_AUDIODRIVER=pulse
```

## 🔄 Restart Services

### Restart Ollama
```bash
pkill -f "ollama serve"
ollama serve &
```

### Restart Application
```bash
# Stop application (Ctrl+C in terminal)
# Then restart
./start_app.sh
```

## 📁 File Structure

After setup, your directory should contain:
```
RealtimeVoiceChat/
├── code/                    # Application source code
├── setup_complete.sh        # Complete setup script
├── setup_ollama.sh         # Ollama-specific setup
├── start_app.sh            # Robust application startup
├── set_audio_env.sh        # Audio environment configuration
├── check_ollama_status.sh  # Ollama status checker
├── requirements.txt        # Python dependencies
└── README.md              # Original project documentation
```

## 🎉 Success Indicators

When everything is working correctly, you should see:
```
🎤🚀 Starting RealtimeVoiceChat Application
=========================================

🦙 Step 1: Checking Ollama service...
✅ Ollama service already running
✅ mistral:7b model available

🔊 Step 2: Setting up audio environment...
✅ Audio environment configured

🎤 Step 3: Checking TTS server...
✅ TTS server already running on port 1234

🧪 Step 4: Running final checks...
✅ All critical packages available

🚀 Step 5: Starting RealtimeVoiceChat application...
🎉 All dependencies ready! Starting application...
```

The application is now ready for robust real-time voice chat!
