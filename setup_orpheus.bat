@echo off
REM setup_orpheus.bat - Automated setup script for Orpheus TTS in RealtimeVoiceChat

echo 🎤🚀 Setting up Orpheus TTS for RealtimeVoiceChat...

REM Configuration
set MODELS_DIR=models
set ORPHEUS_MODEL_PATH=%MODELS_DIR%\Orpheus-3b-FT-Q8_0.gguf
set ORPHEUS_MODEL_URL=https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf

echo 📁 Creating models directory...
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

REM Check if model already exists
if exist "%ORPHEUS_MODEL_PATH%" (
    echo ✅ Orpheus model already exists at %ORPHEUS_MODEL_PATH%
) else (
    echo ⏬ Downloading Orpheus model (this may take a while)...
    echo 📥 Downloading from: %ORPHEUS_MODEL_URL%
    
    REM Try to download using curl (available in Windows 10+)
    curl -L "%ORPHEUS_MODEL_URL%" -o "%ORPHEUS_MODEL_PATH%"
    
    if exist "%ORPHEUS_MODEL_PATH%" (
        echo ✅ Orpheus model downloaded successfully
    ) else (
        echo ❌ Failed to download Orpheus model
        echo 💡 Please manually download the model from:
        echo    %ORPHEUS_MODEL_URL%
        echo    and save it as: %ORPHEUS_MODEL_PATH%
        pause
        exit /b 1
    )
)

REM Install llama-cpp-python with server support and CUDA
echo 🔧 Installing llama-cpp-python with server support and CUDA...

REM Set CUDA compilation flags
set CMAKE_ARGS=-DLLAMA_CUDA=on

REM Install llama-cpp-python with server support
python -m pip install llama-cpp-python[server] --force-reinstall --no-cache-dir

if %ERRORLEVEL% EQU 0 (
    echo ✅ llama-cpp-python[server] installed successfully
) else (
    echo ❌ Failed to install llama-cpp-python[server]
    pause
    exit /b 1
)

REM Test if we can import the server module
echo 🧪 Testing llama-cpp-python server installation...
python -c "import llama_cpp.server; print('✅ llama-cpp-python server module imported successfully')"

if %ERRORLEVEL% EQU 0 (
    echo 🎉 Orpheus TTS setup completed successfully!
    echo 📝 Summary:
    echo    • Model location: %ORPHEUS_MODEL_PATH%
    echo    • llama-cpp-python[server] installed with CUDA support
    echo.
    echo 💡 The application will now automatically start the Orpheus server when needed.
    echo 🚀 You can now run your RealtimeVoiceChat application!
) else (
    echo ❌ Failed to import llama-cpp-python server module
    pause
    exit /b 1
)

pause
