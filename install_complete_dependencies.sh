#!/bin/bash

# install_complete_dependencies.sh - Complete dependency installation for RealtimeVoiceChat on RunPod

set -e  # Exit on any error

echo "🚀 Complete RealtimeVoiceChat Dependency Installation for RunPod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="/workspace/models"
ORPHEUS_MODEL_PATH="${MODELS_DIR}/Orpheus-3b-FT-Q8_0.gguf"
ORPHEUS_MODEL_URL="https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf"

echo -e "${BLUE}📋 Installation Plan:${NC}"
echo -e "   1. Update system packages"
echo -e "   2. Install system audio dependencies"
echo -e "   3. Install PyTorch with CUDA support"
echo -e "   4. Install llama-cpp-python with server support"
echo -e "   5. Install all Python dependencies"
echo -e "   6. Download Orpheus model"
echo -e "   7. Verify installation"
echo ""

# Step 1: Update system packages
echo -e "${YELLOW}📦 Step 1: Updating system packages...${NC}"
apt-get update

# Step 2: Install system audio dependencies
echo -e "${YELLOW}🔊 Step 2: Installing system audio dependencies...${NC}"
apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    ffmpeg \
    git \
    curl \
    wget

echo -e "${GREEN}✅ System dependencies installed${NC}"

# Step 3: Install PyTorch with CUDA support
echo -e "${YELLOW}🔥 Step 3: Installing PyTorch with CUDA 12.1 support...${NC}"
pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo -e "${GREEN}✅ PyTorch with CUDA support installed${NC}"

# Step 4: Install llama-cpp-python with server support and CUDA
echo -e "${YELLOW}🦙 Step 4: Installing llama-cpp-python with server support and CUDA...${NC}"
export CMAKE_ARGS="-DLLAMA_CUDA=on"
pip install --no-cache-dir llama-cpp-python[server] --force-reinstall

echo -e "${GREEN}✅ llama-cpp-python[server] with CUDA support installed${NC}"

# Step 5: Install core scientific computing libraries
echo -e "${YELLOW}🧮 Step 5: Installing core scientific computing libraries...${NC}"
pip install --no-cache-dir \
    numpy \
    scipy \
    transformers \
    huggingface_hub

echo -e "${GREEN}✅ Core scientific libraries installed${NC}"

# Step 6: Install speech processing libraries
echo -e "${YELLOW}🎤 Step 6: Installing speech processing libraries...${NC}"
pip install --no-cache-dir \
    realtimestt==0.3.104 \
    "realtimetts[kokoro,coqui,orpheus]==0.5.7"

echo -e "${GREEN}✅ Speech processing libraries installed${NC}"

# Step 7: Install web server and utility dependencies
echo -e "${YELLOW}🌐 Step 7: Installing web server and utility dependencies...${NC}"
pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-dotenv \
    requests \
    openai \
    ollama

echo -e "${GREEN}✅ Web server and utility dependencies installed${NC}"

# Step 8: Install compatibility constraints
echo -e "${YELLOW}🔧 Step 8: Installing compatibility constraints...${NC}"
pip install --no-cache-dir "ctranslate2<4.5.0"

echo -e "${GREEN}✅ Compatibility constraints applied${NC}"

# Step 9: Create models directory and download Orpheus model
echo -e "${YELLOW}📁 Step 9: Setting up models directory...${NC}"
mkdir -p "${MODELS_DIR}"

if [ -f "${ORPHEUS_MODEL_PATH}" ]; then
    echo -e "${GREEN}✅ Orpheus model already exists at ${ORPHEUS_MODEL_PATH}${NC}"
else
    echo -e "${YELLOW}⏬ Downloading Orpheus model (this may take a while)...${NC}"
    echo -e "${BLUE}📥 Downloading from: ${ORPHEUS_MODEL_URL}${NC}"
    
    if command -v wget &> /dev/null; then
        wget "${ORPHEUS_MODEL_URL}" -O "${ORPHEUS_MODEL_PATH}"
    elif command -v curl &> /dev/null; then
        curl -L "${ORPHEUS_MODEL_URL}" -o "${ORPHEUS_MODEL_PATH}"
    else
        echo -e "${RED}❌ Neither wget nor curl found. Please install one of them.${NC}"
        exit 1
    fi
    
    if [ -f "${ORPHEUS_MODEL_PATH}" ]; then
        echo -e "${GREEN}✅ Orpheus model downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download Orpheus model${NC}"
        exit 1
    fi
fi

# Check model file size (should be around 3GB)
MODEL_SIZE=$(du -h "${ORPHEUS_MODEL_PATH}" | cut -f1)
echo -e "${BLUE}📊 Model size: ${MODEL_SIZE}${NC}"

# Step 10: Verification
echo -e "${YELLOW}🧪 Step 10: Verifying installation...${NC}"

echo -e "${BLUE}Testing core imports...${NC}"
python -c "
import torch
import numpy as np
import scipy
import transformers
import huggingface_hub
print('✅ Core scientific libraries: OK')
"

echo -e "${BLUE}Testing PyTorch CUDA...${NC}"
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA device count: {torch.cuda.device_count()}')
    print(f'✅ Current CUDA device: {torch.cuda.current_device()}')
"

echo -e "${BLUE}Testing speech libraries...${NC}"
python -c "
try:
    from RealtimeSTT import AudioToTextRecorder
    print('✅ RealtimeSTT: OK')
except ImportError as e:
    print(f'❌ RealtimeSTT: {e}')

try:
    from RealtimeTTS import CoquiEngine, KokoroEngine, OrpheusEngine
    print('✅ RealtimeTTS: OK')
except ImportError as e:
    print(f'❌ RealtimeTTS: {e}')
"

echo -e "${BLUE}Testing llama-cpp-python server...${NC}"
python -c "
try:
    import llama_cpp.server
    print('✅ llama-cpp-python server module: OK')
except ImportError as e:
    print(f'❌ llama-cpp-python server: {e}')
"

echo -e "${BLUE}Testing web server dependencies...${NC}"
python -c "
import fastapi
import uvicorn
import requests
print('✅ Web server dependencies: OK')
"

# Get PyTorch version for summary
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

echo -e "${GREEN}🎉 Complete dependency installation finished successfully!${NC}"
echo -e "${BLUE}📝 Installation Summary:${NC}"
echo -e "   • System audio dependencies installed"
echo -e "   • PyTorch ${TORCH_VERSION} with CUDA support"
echo -e "   • llama-cpp-python[server] with CUDA support"
echo -e "   • All scientific computing libraries (numpy, scipy, transformers)"
echo -e "   • Speech processing libraries (RealtimeSTT, RealtimeTTS)"
echo -e "   • Web server dependencies (FastAPI, uvicorn)"
echo -e "   • Orpheus model: ${ORPHEUS_MODEL_PATH} (${MODEL_SIZE})"
echo -e ""
echo -e "${YELLOW}💡 The application should now start without dependency errors.${NC}"
echo -e "${BLUE}🚀 To start the application:${NC}"
echo -e "   cd /workspace/RealtimeVoiceChat/code"
echo -e "   python server.py"
