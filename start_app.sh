#!/bin/bash

# start_app.sh - Robust startup script for RealtimeVoiceChat
# Ensures all dependencies are ready before launching the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🎤🚀 Starting RealtimeVoiceChat Application${NC}"
echo -e "${PURPLE}=========================================${NC}"
echo ""

# Configuration
OLLAMA_PORT=11434
TTS_PORT=1234
OLLAMA_MODEL="mistral:7b"

# Step 1: Check and start Ollama
echo -e "${YELLOW}🦙 Step 1: Checking Ollama service...${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama not found. Please run setup_complete.sh first${NC}"
    exit 1
fi

# Check if Ollama service is running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo -e "${BLUE}🚀 Starting Ollama service...${NC}"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for service to start
    echo -e "${BLUE}⏳ Waiting for Ollama service...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama service started${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Ollama service failed to start${NC}"
            echo -e "${BLUE}📋 Ollama log:${NC}"
            tail -10 /tmp/ollama.log
            exit 1
        fi
        
        sleep 1
    done
else
    echo -e "${GREEN}✅ Ollama service already running${NC}"
fi

# Check if model is available
echo -e "${BLUE}🔍 Checking Ollama model...${NC}"
if ollama list | grep -q "${OLLAMA_MODEL}"; then
    echo -e "${GREEN}✅ ${OLLAMA_MODEL} model available${NC}"
else
    echo -e "${YELLOW}⏬ Downloading ${OLLAMA_MODEL} model...${NC}"
    ollama pull ${OLLAMA_MODEL}
    
    if ollama list | grep -q "${OLLAMA_MODEL}"; then
        echo -e "${GREEN}✅ ${OLLAMA_MODEL} model downloaded${NC}"
    else
        echo -e "${RED}❌ Failed to download ${OLLAMA_MODEL} model${NC}"
        exit 1
    fi
fi

# Step 2: Set audio environment
echo -e "${YELLOW}🔊 Step 2: Setting up audio environment...${NC}"

# Source audio environment if available
if [ -f "set_audio_env.sh" ]; then
    source set_audio_env.sh
    echo -e "${GREEN}✅ Audio environment configured${NC}"
else
    echo -e "${BLUE}🔧 Setting basic audio environment...${NC}"
    export ALSA_PCM_CARD=default
    export ALSA_PCM_DEVICE=0
    export PULSE_RUNTIME_PATH=/tmp/pulse-runtime
    export SDL_AUDIODRIVER=pulse
    echo -e "${GREEN}✅ Basic audio environment set${NC}"
fi

# Step 3: Check TTS server (optional)
echo -e "${YELLOW}🎤 Step 3: Checking TTS server...${NC}"

if curl -s http://localhost:${TTS_PORT}/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ TTS server already running on port ${TTS_PORT}${NC}"
elif netstat -tuln | grep -q ":${TTS_PORT} "; then
    echo -e "${GREEN}✅ Port ${TTS_PORT} is in use (assuming TTS server)${NC}"
else
    echo -e "${YELLOW}⚠️ TTS server not running on port ${TTS_PORT}${NC}"
    echo -e "${BLUE}💡 TTS will be started automatically or you can start manually:${NC}"
    echo -e "${BLUE}   python -m llama_cpp.server --model /workspace/models/Orpheus-3b-FT-Q8_0.gguf --host 0.0.0.0 --port ${TTS_PORT} --n_gpu_layers -1${NC}"
fi

# Step 4: Final checks
echo -e "${YELLOW}🧪 Step 4: Running final checks...${NC}"

# Check Python dependencies
echo -e "${BLUE}🔍 Checking Python dependencies...${NC}"
cd code

python3 -c "
import sys
critical_packages = ['fastapi', 'uvicorn', 'whisper', 'torch', 'requests']
missing = []

for package in critical_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    print('Please run: pip install -r ../requirements.txt')
    sys.exit(1)
else:
    print('✅ All critical packages available')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python dependency check failed${NC}"
    exit 1
fi

# Step 5: Start the application
echo -e "${YELLOW}🚀 Step 5: Starting RealtimeVoiceChat application...${NC}"
echo ""
echo -e "${GREEN}🎉 All dependencies ready! Starting application...${NC}"
echo -e "${BLUE}📝 Configuration:${NC}"
echo -e "   • Ollama: ✅ Running on port ${OLLAMA_PORT}"
echo -e "   • Model: ✅ ${OLLAMA_MODEL}"
echo -e "   • Audio: ✅ Environment configured"
echo -e "   • TTS: 🔄 Will start automatically or use manual server"
echo ""
echo -e "${PURPLE}🎤 Starting RealtimeVoiceChat server...${NC}"
echo -e "${PURPLE}====================================${NC}"

# Start the application
exec python3 server.py
