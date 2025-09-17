#!/usr/bin/env python3
"""
test_orpheus_server.py - Test Orpheus server startup independently

This script tests starting the Orpheus server in the background and verifying it works.
"""

import os
import sys
import subprocess
import time
import requests
import signal

def test_server_startup():
    """Test starting the Orpheus server manually."""
    model_path = "/workspace/models/Orpheus-3b-FT-Q8_0.gguf"
    port = 1234
    
    print("🧪 Testing Orpheus Server Startup")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"✅ Model file found: {size_gb:.1f} GB")
    
    # Check if port is already in use
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ Server already running on port {port}")
            return True
    except:
        pass
    
    # Start server
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n_gpu_layers", "-1"
    ]
    
    print(f"🚀 Starting server with command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        print(f"🎤 Server process started (PID: {process.pid})")
        
        # Wait for server to start
        start_time = time.time()
        timeout = 180  # 3 minutes
        
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            
            # Check if process died
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Server process died after {elapsed:.1f}s")
                if stdout.strip():
                    print(f"📤 stdout: {stdout}")
                if stderr.strip():
                    print(f"📤 stderr: {stderr}")
                return False
            
            # Check if server is responding
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
                if response.status_code == 200:
                    print(f"✅ Server started successfully in {elapsed:.1f}s")
                    print(f"🎉 Server is responding on port {port}")
                    
                    # Test a simple completion
                    try:
                        test_data = {
                            "prompt": "Hello",
                            "max_tokens": 5,
                            "temperature": 0.7
                        }
                        response = requests.post(
                            f"http://127.0.0.1:{port}/v1/completions",
                            json=test_data,
                            timeout=10
                        )
                        if response.status_code == 200:
                            print("✅ Server API test successful")
                        else:
                            print(f"⚠️  Server API test failed: {response.status_code}")
                    except Exception as e:
                        print(f"⚠️  Server API test error: {e}")
                    
                    # Keep server running for a bit
                    print("🎤 Server is running. Press Ctrl+C to stop...")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping server...")
                        
                    # Clean up
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=10)
                        print("✅ Server stopped cleanly")
                    except:
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            print("⚠️  Server force killed")
                        except:
                            print("❌ Could not stop server")
                    
                    return True
                    
            except requests.exceptions.RequestException:
                pass
            
            # Log progress every 30 seconds
            if int(elapsed) % 30 == 0 and elapsed > 0:
                print(f"🎤⏳ Waiting for server... ({elapsed:.0f}s/{timeout}s)")
            
            time.sleep(3)
        
        print(f"❌ Server failed to start within {timeout} seconds")
        
        # Try to get output
        try:
            stdout, stderr = process.communicate(timeout=5)
            if stdout.strip():
                print(f"📤 stdout: {stdout}")
            if stderr.strip():
                print(f"📤 stderr: {stderr}")
        except:
            print("⚠️  Could not get server output")
        
        # Clean up
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
        
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1)
