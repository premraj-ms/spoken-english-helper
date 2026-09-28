# 🚀 Complete Google Colab GPU Setup Guide for Elena AI Spoken English Coach

Run the entire pipeline (Whisper Speech-to-Text, Ollama LLM Reasoning, and Kokoro Voice Synthesis) on **Google Colab's Free T4 GPU** with ultra-low latency.

---

### ⚙️ Step 0: Set Colab Runtime to GPU
1. In Google Colab, go to the menu: **Runtime** -> **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or A100/V100 if available).
3. Click **Save**.

---

### 📦 Step 1: Clone Repository & Install GPU Dependencies
Run this in a Colab code cell:
```python
# 1. Clone your GitHub repository
!git clone https://github.com/premraj-ms/spoken-english-helper.git
%cd spoken-english-helper

# 2. Install all required dependencies with CUDA support
!pip install -q fastapi uvicorn python-multipart faster-whisper kokoro-onnx soundfile reportlab ollama
!pip install -q onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || pip install -q onnxruntime

# 3. Create models directory and download Kokoro TTS weights
!mkdir -p models
!wget -q -nc -O models/kokoro-v0_19.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v0_19.onnx
!wget -q -nc -O models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
print("✅ Environment & Model Weights Ready!")
```

---

### 🦙 Step 2: Install & Start Ollama on GPU
Ollama will automatically bind to the Colab GPU (CUDA):
```python
# 1. Install Ollama Linux binary
!curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama daemon in background
import subprocess
import time

ollama_proc = subprocess.Popen(["ollama", "serve"])
time.sleep(4)

# 3. Pull fast model (llama3.2 or qwen2.5:7b runs lightning fast on T4 GPU)
!ollama pull llama3.2
# Alternatively: !ollama pull qwen2.5:7b
print("✅ Ollama is running on GPU!")
```

---

### 🌐 Step 3: Install Cloudflare Tunnel (HTTPS for Microphone Access)
Browsers require **HTTPS** for microphone access. Cloudflare Tunnel provides an instant, free, secure HTTPS URL:
```python
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
print("✅ Cloudflared Installed!")
```

---

### 🎙️ Step 4: Launch Full Application & Public Web URL
```python
import subprocess
import threading
import time

# 1. Start FastAPI backend in background thread
def start_app():
    subprocess.run(["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"])

server_thread = threading.Thread(target=start_app, daemon=True)
server_thread.start()
time.sleep(3)

# 2. Expose through Cloudflare HTTPS Tunnel
print("\n" + "="*70)
print("🎙️ CLICK THE 'trycloudflare.com' HTTPS LINK BELOW TO START SPEAKING:")
print("="*70 + "\n")

!cloudflared tunnel --url http://localhost:8000
```

---

### ⚡ GPU Acceleration Breakdown:
| Component | Engine | GPU Acceleration Mode | Latency on Colab T4 |
|---|---|---|---|
| **STT (Speech-to-Text)** | Faster-Whisper | `device="cuda"`, `float16` | **~50 - 100 ms** |
| **LLM Reasoning & Feedback** | Ollama (`llama3.2` / `qwen2.5`) | NVIDIA CUDA VRAM | **~0.3 - 0.8 s** |
| **TTS (Voice Synthesis)** | Kokoro ONNX | ONNX Runtime Engine | **~100 - 200 ms** |

