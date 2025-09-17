@echo off
REM setup_complete.bat - Complete setup for RealtimeVoiceChat on Windows

echo.
echo 🎤🚀 RealtimeVoiceChat Complete Setup for Windows
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    echo 💡 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found. Please ensure pip is installed with Python.
    pause
    exit /b 1
)

echo ✅ pip found

REM Install Python dependencies
echo.
echo 🐍 Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo ✅ Python dependencies installed

REM Check if Ollama is installed
echo.
echo 🦙 Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not found. Please install Ollama first.
    echo 💡 Download from: https://ollama.ai/download
    echo 💡 After installation, run: ollama pull mistral:7b
    pause
    exit /b 1
)

echo ✅ Ollama found
ollama --version

REM Check if Ollama service is running
echo.
echo 🔍 Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Ollama service not responding
    echo 💡 Please start Ollama service and run: ollama pull mistral:7b
    echo 💡 Then restart this setup script
    pause
    exit /b 1
)

echo ✅ Ollama service is running

REM Check if Mistral model is available
echo.
echo 🤖 Checking Mistral model...
ollama list | findstr "mistral" >nul 2>&1
if errorlevel 1 (
    echo ⏬ Downloading Mistral model...
    ollama pull mistral:7b
    if errorlevel 1 (
        echo ❌ Failed to download Mistral model
        pause
        exit /b 1
    )
)

echo ✅ Mistral model available

REM Test Python imports
echo.
echo 🧪 Testing Python imports...
python -c "import whisper, torch, numpy, scipy, transformers, requests, fastapi, uvicorn; print('✅ All imports successful')"
if errorlevel 1 (
    echo ❌ Some Python packages are missing
    echo 💡 Try: pip install -r requirements.txt --force-reinstall
    pause
    exit /b 1
)

REM Create startup batch file
echo.
echo 📝 Creating startup script...
(
echo @echo off
echo echo 🎤🚀 Starting RealtimeVoiceChat Application
echo echo =========================================
echo echo.
echo.
echo REM Check Ollama service
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo ❌ Ollama service not running
echo     echo 💡 Please start Ollama service first
echo     pause
echo     exit /b 1
echo ^)
echo echo ✅ Ollama service is running
echo.
echo REM Start the application
echo echo 🚀 Starting RealtimeVoiceChat server...
echo cd code
echo python server.py
) > start_app.bat

echo ✅ Created start_app.bat

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📝 Summary:
echo    • Python dependencies: ✅ Installed
echo    • Ollama service: ✅ Running
echo    • Mistral model: ✅ Available
echo    • Startup script: ✅ Created
echo.
echo 🚀 To start the application:
echo    1. Double-click start_app.bat
echo    2. Or run: start_app.bat
echo.
echo 💡 Manual TTS server (if needed):
echo    python -m llama_cpp.server --model /workspace/models/Orpheus-3b-FT-Q8_0.gguf --host 0.0.0.0 --port 1234 --n_gpu_layers -1
echo.
pause
