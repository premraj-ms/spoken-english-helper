# Google Colab Setup & Runner for Elena AI Spoken English Coach

## 📌 Requirements for Colab:
- Set Runtime Type: **GPU (T4)** (for ultra-fast Whisper STT and Ollama) or **CPU**.
- Microphone access requires **HTTPS**, which is automatically handled by the **Cloudflare Tunnel** below.

---

### Step 1: Install Dependencies
```python
# 1. Clone repository or download project files
# !git clone <YOUR_REPO_URL>
# %cd "English Communication"

# 2. Install Python packages
!pip install -q fastapi uvicorn python-multipart faster-whisper kokoro-onnx soundfile reportlab ollama
```

### Step 2: Install and Start Ollama in Colab
```python
# Install Ollama binary
!curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama background service in Colab
import subprocess
import time

ollama_process = subprocess.Popen(["ollama", "serve"])
time.sleep(4)

# Pull your preferred model (e.g., gpt-oss:120b-cloud, llama3.2, or qwen2.5:7b)
# Note: In Colab GPU, llama3.2:3b or qwen2.5:7b runs in seconds!
!ollama pull gpt-oss:120b-cloud
```

### Step 3: Install Cloudflared Tunnel (for Free HTTPS Web Access with Mic Support)
```python
# Download cloudflared for public HTTPS URL (enables WebRTC/Microphone access in browser)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

### Step 4: Launch FastAPI Server & Start Cloudflare Tunnel
```python
import subprocess
import threading
import time

def run_server():
    subprocess.run(["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"])

# Start server in background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

# Start Cloudflare tunnel and print public HTTPS URL
print("\n" + "="*60)
print("🚀 LAUNCHING PUBLIC HTTPS TUNNEL FOR MIC ACCESS...")
print("="*60 + "\n")

!cloudflared tunnel --url http://localhost:8000
```
