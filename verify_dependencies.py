#!/usr/bin/env python3
"""
verify_dependencies.py - Comprehensive dependency verification for RealtimeVoiceChat

This script verifies that all required dependencies are properly installed
and can be imported without errors.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Dict, Any

def test_import(module_name: str, from_module: str = None, alias: str = None) -> Tuple[bool, str]:
    """
    Test if a module can be imported successfully.
    
    Args:
        module_name: Name of the module to import
        from_module: If specified, import from this module
        alias: If specified, import as this alias
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        if from_module:
            if alias:
                exec(f"from {from_module} import {module_name} as {alias}")
            else:
                exec(f"from {from_module} import {module_name}")
        else:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def check_pytorch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA availability and configuration."""
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None
        }
    except ImportError:
        return {"error": "PyTorch not installed"}

def check_model_file(model_path: str) -> Dict[str, Any]:
    """Check if the Orpheus model file exists and get its size."""
    import os
    try:
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_gb = size_bytes / (1024**3)
            return {
                "exists": True,
                "size_bytes": size_bytes,
                "size_gb": round(size_gb, 2),
                "path": model_path
            }
        else:
            return {"exists": False, "path": model_path}
    except Exception as e:
        return {"exists": False, "error": str(e), "path": model_path}

def main():
    """Main verification function."""
    print("🧪 RealtimeVoiceChat Dependency Verification")
    print("=" * 50)
    
    # Define all required imports
    core_imports = [
        ("numpy", None, "np"),
        ("scipy", None, None),
        ("torch", None, None),
        ("transformers", None, None),
        ("huggingface_hub", None, None),
    ]
    
    speech_imports = [
        ("AudioToTextRecorder", "RealtimeSTT", None),
        ("CoquiEngine", "RealtimeTTS", None),
        ("KokoroEngine", "RealtimeTTS", None),
        ("OrpheusEngine", "RealtimeTTS", None),
        ("TextToAudioStream", "RealtimeTTS", None),
    ]
    
    web_imports = [
        ("fastapi", None, None),
        ("uvicorn", None, None),
        ("requests", None, None),
    ]
    
    llm_imports = [
        ("openai", None, None),
        ("ollama", None, None),
    ]
    
    server_imports = [
        ("llama_cpp.server", None, None),
    ]
    
    audio_processing_imports = [
        ("resample_poly", "scipy.signal", None),
        ("signal", "scipy", None),
    ]
    
    all_tests = [
        ("Core Scientific Libraries", core_imports),
        ("Speech Processing Libraries", speech_imports),
        ("Web Server Dependencies", web_imports),
        ("LLM Provider Libraries", llm_imports),
        ("Server Dependencies", server_imports),
        ("Audio Processing", audio_processing_imports),
    ]
    
    # Track results
    total_tests = 0
    passed_tests = 0
    failed_imports = []
    
    # Run import tests
    for category, imports in all_tests:
        print(f"\n📦 {category}")
        print("-" * 30)
        
        for module_name, from_module, alias in imports:
            total_tests += 1
            success, error = test_import(module_name, from_module, alias)
            
            if success:
                passed_tests += 1
                if from_module:
                    print(f"✅ from {from_module} import {module_name}")
                else:
                    print(f"✅ import {module_name}")
            else:
                failed_imports.append((category, module_name, from_module, error))
                if from_module:
                    print(f"❌ from {from_module} import {module_name} - {error}")
                else:
                    print(f"❌ import {module_name} - {error}")
    
    # Check PyTorch CUDA
    print(f"\n🔥 PyTorch CUDA Configuration")
    print("-" * 30)
    cuda_info = check_pytorch_cuda()
    if "error" in cuda_info:
        print(f"❌ {cuda_info['error']}")
    else:
        print(f"✅ PyTorch version: {cuda_info['version']}")
        print(f"✅ CUDA available: {cuda_info['cuda_available']}")
        if cuda_info['cuda_available']:
            print(f"✅ CUDA device count: {cuda_info['cuda_device_count']}")
            print(f"✅ Current CUDA device: {cuda_info['current_device']}")
            if cuda_info['cuda_version']:
                print(f"✅ CUDA version: {cuda_info['cuda_version']}")
        else:
            print("⚠️  CUDA not available - GPU acceleration disabled")
    
    # Check Orpheus model
    print(f"\n🎤 Orpheus Model File")
    print("-" * 30)
    model_path = "/workspace/models/Orpheus-3b-FT-Q8_0.gguf"
    model_info = check_model_file(model_path)
    
    if model_info["exists"]:
        print(f"✅ Model file exists: {model_info['path']}")
        print(f"✅ Model size: {model_info['size_gb']} GB ({model_info['size_bytes']:,} bytes)")
        if model_info['size_gb'] < 2.5:
            print("⚠️  Model file seems small - may be incomplete")
    else:
        print(f"❌ Model file not found: {model_info['path']}")
        if "error" in model_info:
            print(f"   Error: {model_info['error']}")
    
    # Summary
    print(f"\n📊 Verification Summary")
    print("=" * 30)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_imports)}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_imports:
        print(f"\n❌ Failed Imports:")
        for category, module, from_module, error in failed_imports:
            if from_module:
                print(f"   {category}: from {from_module} import {module}")
            else:
                print(f"   {category}: import {module}")
            print(f"      Error: {error}")
    
    # Recommendations
    if failed_imports:
        print(f"\n💡 Recommendations:")
        print("   1. Run the complete installation script:")
        print("      chmod +x install_complete_dependencies.sh")
        print("      ./install_complete_dependencies.sh")
        print("   2. If specific packages fail, install manually:")
        print("      pip install <package_name>")
        print("   3. For CUDA issues, ensure NVIDIA drivers are installed")
        
        return 1  # Exit with error code
    else:
        print(f"\n🎉 All dependencies verified successfully!")
        print("   Your RealtimeVoiceChat installation is ready to run.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
