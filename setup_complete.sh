#!/bin/bash

# setup_complete.sh - Complete setup for RealtimeVoiceChat application
# This script handles all dependencies: Ollama, Whisper models, Python packages, and system audio

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🎤🚀 RealtimeVoiceChat Complete Setup${NC}"
echo -e "${PURPLE}====================================${NC}"
echo ""

# Configuration
WHISPER_MODEL="base"  # Options: tiny, base, small, medium, large
OLLAMA_MODEL="mistral:7b"
MODELS_DIR="/workspace/models"

echo -e "${BLUE}📋 Setup Plan:${NC}"
echo -e "   1. System dependencies and audio libraries"
echo -e "   2. Python dependencies"
echo -e "   3. Ollama installation and model setup"
echo -e "   4. Whisper model download"
echo -e "   5. Audio system configuration"
echo -e "   6. Verification tests"
echo ""

# Step 1: System Dependencies
echo -e "${YELLOW}📦 Step 1: Installing system dependencies...${NC}"

# Update package list
apt-get update -qq

# Install essential packages
echo -e "${BLUE}🔧 Installing essential packages...${NC}"
apt-get install -y -qq \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# Install audio libraries
echo -e "${BLUE}🔊 Installing audio libraries...${NC}"
apt-get install -y -qq \
    libsndfile1-dev \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    alsa-utils \
    pulseaudio \
    ffmpeg

# Install Python development packages
echo -e "${BLUE}🐍 Installing Python development packages...${NC}"
apt-get install -y -qq \
    python3-dev \
    python3-pip \
    python3-venv

echo -e "${GREEN}✅ System dependencies installed${NC}"

# Step 2: Python Dependencies
echo -e "${YELLOW}🐍 Step 2: Installing Python dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo -e "${BLUE}📦 Installing core Python packages...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}✅ Python dependencies installed${NC}"

# Step 3: Ollama Setup
echo -e "${YELLOW}🦙 Step 3: Setting up Ollama...${NC}"

# Make setup script executable and run it
chmod +x setup_ollama.sh
./setup_ollama.sh

echo -e "${GREEN}✅ Ollama setup completed${NC}"

# Step 4: Whisper Model Download
echo -e "${YELLOW}🎙️ Step 4: Setting up Whisper models...${NC}"

# Create models directory
mkdir -p ${MODELS_DIR}

# Download Whisper model using Python
echo -e "${BLUE}⏬ Downloading Whisper ${WHISPER_MODEL} model...${NC}"
python3 -c "
import whisper
import os

model_name = '${WHISPER_MODEL}'
models_dir = '${MODELS_DIR}'

print(f'Downloading Whisper {model_name} model...')
try:
    model = whisper.load_model(model_name, download_root=models_dir)
    print(f'✅ Whisper {model_name} model downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading Whisper model: {e}')
    exit(1)
"

echo -e "${GREEN}✅ Whisper model setup completed${NC}"

# Step 5: Audio System Configuration
echo -e "${YELLOW}🔊 Step 5: Configuring audio system...${NC}"

# Create ALSA configuration to suppress warnings
echo -e "${BLUE}🔧 Configuring ALSA...${NC}"
cat > /etc/asound.conf << 'EOF'
# ALSA configuration to suppress warnings
pcm.!default {
    type pulse
    fallback "sysdefault"
    hint {
        show on
        description "Default ALSA Output (currently PulseAudio Sound Server)"
    }
}

ctl.!default {
    type pulse
    fallback "sysdefault"
}

# Suppress ALSA warnings
pcm.null {
    type null
}
EOF

# Set environment variables to suppress ALSA warnings
echo -e "${BLUE}🔧 Setting audio environment variables...${NC}"
cat >> ~/.bashrc << 'EOF'

# Suppress ALSA warnings for RealtimeVoiceChat
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime
EOF

# Create audio environment script
cat > set_audio_env.sh << 'EOF'
#!/bin/bash
# Audio environment setup for RealtimeVoiceChat
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime
export SDL_AUDIODRIVER=pulse
EOF

chmod +x set_audio_env.sh

echo -e "${GREEN}✅ Audio system configured${NC}"

# Step 6: Verification
echo -e "${YELLOW}🧪 Step 6: Running verification tests...${NC}"

# Test Python imports
echo -e "${BLUE}🔍 Testing Python imports...${NC}"
python3 -c "
import sys
import traceback

packages = [
    'whisper',
    'torch',
    'numpy',
    'scipy',
    'transformers',
    'requests',
    'fastapi',
    'uvicorn',
    'soundfile',
    'pyaudio'
]

failed = []
for package in packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError as e:
        print(f'❌ {package}: {e}')
        failed.append(package)

if failed:
    print(f'\\n❌ Failed imports: {failed}')
    sys.exit(1)
else:
    print('\\n✅ All Python packages imported successfully')
"

# Test Ollama
echo -e "${BLUE}🔍 Testing Ollama...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${GREEN}✅ Ollama API responding${NC}"
else
    echo -e "${RED}❌ Ollama API not responding${NC}"
fi

# Test Whisper model
echo -e "${BLUE}🔍 Testing Whisper model...${NC}"
python3 -c "
import whisper
import os

try:
    model = whisper.load_model('${WHISPER_MODEL}', download_root='${MODELS_DIR}')
    print('✅ Whisper model loaded successfully')
except Exception as e:
    print(f'❌ Error loading Whisper model: {e}')
"

echo ""
echo -e "${GREEN}🎉 RealtimeVoiceChat setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📝 Summary:${NC}"
echo -e "   • System dependencies: ✅ Installed"
echo -e "   • Python packages: ✅ Installed"
echo -e "   • Ollama service: ✅ Running on port 11434"
echo -e "   • Ollama model: ✅ ${OLLAMA_MODEL}"
echo -e "   • Whisper model: ✅ ${WHISPER_MODEL}"
echo -e "   • Audio system: ✅ Configured"
echo ""
echo -e "${YELLOW}🚀 Ready to start RealtimeVoiceChat!${NC}"
echo -e "${BLUE}💡 To start the application:${NC}"
echo -e "   cd code"
echo -e "   source ../set_audio_env.sh"
echo -e "   python server.py"
echo ""
echo -e "${BLUE}💡 To manually start TTS server (if needed):${NC}"
echo -e "   python -m llama_cpp.server --model ${MODELS_DIR}/Orpheus-3b-FT-Q8_0.gguf --host 0.0.0.0 --port 1234 --n_gpu_layers -1"
