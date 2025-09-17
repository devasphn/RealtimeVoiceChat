#!/bin/bash

# setup_ollama.sh - Automatic Ollama installation and model setup for RealtimeVoiceChat

set -e  # Exit on any error

echo "🦙 Setting up Ollama for RealtimeVoiceChat..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_MODEL="mistral:7b"  # Smaller, faster model than the original
OLLAMA_PORT=11434

echo -e "${BLUE}📋 Ollama Setup Plan:${NC}"
echo -e "   1. Install Ollama"
echo -e "   2. Start Ollama service"
echo -e "   3. Download Mistral model"
echo -e "   4. Verify installation"
echo ""

# Step 1: Install Ollama
echo -e "${YELLOW}📦 Step 1: Installing Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama already installed${NC}"
    ollama --version
else
    echo -e "${BLUE}⏬ Downloading and installing Ollama...${NC}"
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✅ Ollama installed successfully${NC}"
        ollama --version
    else
        echo -e "${RED}❌ Failed to install Ollama${NC}"
        exit 1
    fi
fi

# Step 2: Start Ollama service
echo -e "${YELLOW}🚀 Step 2: Starting Ollama service...${NC}"

# Check if Ollama is already running
if pgrep -f "ollama serve" > /dev/null; then
    echo -e "${GREEN}✅ Ollama service already running${NC}"
else
    echo -e "${BLUE}🔧 Starting Ollama service...${NC}"
    
    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for service to start
    echo -e "${BLUE}⏳ Waiting for Ollama service to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama service started successfully${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Ollama service failed to start within 30 seconds${NC}"
            echo -e "${BLUE}📋 Ollama log:${NC}"
            tail -20 /tmp/ollama.log
            exit 1
        fi
        
        sleep 1
    done
fi

# Step 3: Download Mistral model
echo -e "${YELLOW}🤖 Step 3: Setting up Mistral model...${NC}"

# Check if model is already available
if ollama list | grep -q "${OLLAMA_MODEL}"; then
    echo -e "${GREEN}✅ ${OLLAMA_MODEL} model already available${NC}"
else
    echo -e "${BLUE}⏬ Downloading ${OLLAMA_MODEL} model (this may take a while)...${NC}"
    
    # Download the model
    ollama pull ${OLLAMA_MODEL}
    
    if ollama list | grep -q "${OLLAMA_MODEL}"; then
        echo -e "${GREEN}✅ ${OLLAMA_MODEL} model downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download ${OLLAMA_MODEL} model${NC}"
        exit 1
    fi
fi

# Step 4: Verify installation
echo -e "${YELLOW}🧪 Step 4: Verifying Ollama installation...${NC}"

# Test API endpoint
echo -e "${BLUE}🔍 Testing Ollama API...${NC}"
if curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null; then
    echo -e "${GREEN}✅ Ollama API responding${NC}"
else
    echo -e "${RED}❌ Ollama API not responding${NC}"
    exit 1
fi

# Test model generation
echo -e "${BLUE}🧪 Testing model generation...${NC}"
TEST_RESPONSE=$(curl -s -X POST http://localhost:${OLLAMA_PORT}/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"Hello\", \"stream\": false}" \
    --max-time 30)

if echo "$TEST_RESPONSE" | grep -q "response"; then
    echo -e "${GREEN}✅ Model generation test successful${NC}"
else
    echo -e "${RED}❌ Model generation test failed${NC}"
    echo -e "${BLUE}Response: ${TEST_RESPONSE}${NC}"
    exit 1
fi

# List available models
echo -e "${BLUE}📋 Available models:${NC}"
ollama list

echo -e "${GREEN}🎉 Ollama setup completed successfully!${NC}"
echo -e "${BLUE}📝 Summary:${NC}"
echo -e "   • Ollama service running on port ${OLLAMA_PORT}"
echo -e "   • Model: ${OLLAMA_MODEL}"
echo -e "   • API endpoint: http://localhost:${OLLAMA_PORT}"
echo -e ""
echo -e "${YELLOW}💡 Ollama is now ready for RealtimeVoiceChat!${NC}"

# Create a simple status check script
cat > check_ollama_status.sh << 'EOF'
#!/bin/bash
echo "🦙 Ollama Status Check"
echo "===================="

if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama service: RUNNING"
else
    echo "❌ Ollama service: NOT RUNNING"
    echo "💡 Start with: ollama serve &"
fi

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama API: RESPONDING"
else
    echo "❌ Ollama API: NOT RESPONDING"
fi

echo ""
echo "📋 Available models:"
ollama list 2>/dev/null || echo "❌ Could not list models"
EOF

chmod +x check_ollama_status.sh
echo -e "${BLUE}💡 Created check_ollama_status.sh for future status checks${NC}"
