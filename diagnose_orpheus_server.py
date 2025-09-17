#!/usr/bin/env python3
"""
diagnose_orpheus_server.py - Diagnostic tool for Orpheus server startup issues

This script helps diagnose why the Orpheus server might be failing to start.
"""

import os
import sys
import subprocess
import socket
import time
import requests
from pathlib import Path

def check_model_file(model_path: str):
    """Check if the Orpheus model file exists and is valid."""
    print(f"🔍 Checking model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        return False
    
    size_bytes = os.path.getsize(model_path)
    size_gb = size_bytes / (1024**3)
    
    print(f"✅ Model file exists")
    print(f"📊 Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    
    if size_gb < 2.5:
        print(f"⚠️  Model file seems small (expected ~3GB)")
        return False
    
    return True

def check_port_availability(port: int):
    """Check if the port is available."""
    print(f"🔍 Checking port {port} availability...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            if result == 0:
                print(f"❌ Port {port} is already in use")
                
                # Try to find what's using the port
                try:
                    result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                          capture_output=True, text=True)
                    if result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        print(f"🔍 Processes using port {port}: {', '.join(pids)}")
                        
                        for pid in pids:
                            try:
                                proc_result = subprocess.run(['ps', '-p', pid, '-o', 'comm='], 
                                                           capture_output=True, text=True)
                                if proc_result.stdout.strip():
                                    print(f"   PID {pid}: {proc_result.stdout.strip()}")
                            except:
                                pass
                except:
                    print(f"💡 To kill processes on port {port}: lsof -ti:{port} | xargs kill -9")
                
                return False
            else:
                print(f"✅ Port {port} is available")
                return True
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def check_llama_cpp_installation():
    """Check if llama-cpp-python is properly installed."""
    print("🔍 Checking llama-cpp-python installation...")
    
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python version: {llama_cpp.__version__}")
    except ImportError:
        print("❌ llama-cpp-python not installed")
        return False
    
    try:
        import llama_cpp.server
        print("✅ llama-cpp-python server module available")
    except ImportError:
        print("❌ llama-cpp-python server module not available")
        print("💡 Install with: pip install llama-cpp-python[server]")
        return False
    
    return True

def test_server_startup(model_path: str, port: int = 1234, timeout: int = 30):
    """Test starting the server manually with detailed output."""
    print(f"🧪 Testing server startup manually...")
    
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n_gpu_layers", "-1"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        print(f"🚀 Starting server (timeout: {timeout}s)...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        
        # Monitor for a short time
        while time.time() - start_time < timeout:
            # Check if process died
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Server process died after {time.time() - start_time:.1f}s")
                print(f"📤 stdout: {stdout}")
                print(f"📤 stderr: {stderr}")
                return False
            
            # Check if server is responding
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"✅ Server started successfully in {elapsed:.1f}s")
                    
                    # Clean up
                    process.terminate()
                    process.wait(timeout=5)
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        # Timeout reached
        print(f"❌ Server did not start within {timeout}s")
        
        # Get output
        try:
            stdout, stderr = process.communicate(timeout=5)
            print(f"📤 stdout: {stdout}")
            print(f"📤 stderr: {stderr}")
        except subprocess.TimeoutExpired:
            print("⚠️  Could not get server output")
            process.kill()
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing server startup: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability for CUDA acceleration."""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
        else:
            print("⚠️  CUDA not available - server will use CPU (slower)")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True

def main():
    """Main diagnostic function."""
    print("🔧 Orpheus Server Diagnostic Tool")
    print("=" * 50)
    
    model_path = "/workspace/models/Orpheus-3b-FT-Q8_0.gguf"
    port = 1234
    
    # Run all checks
    checks = [
        ("Model File", lambda: check_model_file(model_path)),
        ("Port Availability", lambda: check_port_availability(port)),
        ("llama-cpp-python Installation", check_llama_cpp_installation),
        ("GPU Availability", check_gpu_availability),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        results[name] = check_func()
    
    # Summary
    print(f"\n📊 Diagnostic Summary")
    print("=" * 30)
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    # If basic checks pass, test server startup
    if all(results.values()):
        print(f"\n🧪 All basic checks passed. Testing server startup...")
        print("=" * 50)
        test_result = test_server_startup(model_path, port, timeout=60)
        
        if test_result:
            print(f"\n🎉 Server startup test PASSED!")
            print("💡 The Orpheus server should work correctly.")
        else:
            print(f"\n❌ Server startup test FAILED!")
            print("💡 Check the error messages above for details.")
    else:
        print(f"\n⚠️  Some basic checks failed. Fix these issues first:")
        for name, result in results.items():
            if not result:
                print(f"   • {name}")

if __name__ == "__main__":
    main()
