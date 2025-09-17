#!/bin/bash

# setup_orpheus.sh - Automated setup script for Orpheus TTS in RealtimeVoiceChat

set -e  # Exit on any error

echo "🎤🚀 Setting up Orpheus TTS for RealtimeVoiceChat..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="/workspace/models"
ORPHEUS_MODEL_PATH="${MODELS_DIR}/Orpheus-3b-FT-Q8_0.gguf"
ORPHEUS_MODEL_URL="https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf"

echo -e "${BLUE}📁 Creating models directory...${NC}"
mkdir -p "${MODELS_DIR}"

# Check if model already exists
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

# Install llama-cpp-python with server support and CUDA
echo -e "${YELLOW}🔧 Installing llama-cpp-python with server support and CUDA...${NC}"

# Set CUDA compilation flags
export CMAKE_ARGS="-DLLAMA_CUDA=on"

# Install llama-cpp-python with server support
python -m pip install llama-cpp-python[server] --force-reinstall --no-cache-dir

echo -e "${GREEN}✅ llama-cpp-python[server] installed successfully${NC}"

# Test if we can import the server module
echo -e "${BLUE}🧪 Testing llama-cpp-python server installation...${NC}"
python -c "import llama_cpp.server; print('✅ llama-cpp-python server module imported successfully')" || {
    echo -e "${RED}❌ Failed to import llama-cpp-python server module${NC}"
    exit 1
}

echo -e "${GREEN}🎉 Orpheus TTS setup completed successfully!${NC}"
echo -e "${BLUE}📝 Summary:${NC}"
echo -e "   • Model location: ${ORPHEUS_MODEL_PATH}"
echo -e "   • Model size: ${MODEL_SIZE}"
echo -e "   • llama-cpp-python[server] installed with CUDA support"
echo -e ""
echo -e "${YELLOW}💡 The application will now automatically start the Orpheus server when needed.${NC}"
echo -e "${BLUE}🚀 You can now run your RealtimeVoiceChat application!${NC}"
