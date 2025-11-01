<div align="center">
  <img src="https://raw.githubusercontent.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
  <h1>microWakeWord Trainer – NVIDIA Docker Edition</h1>
</div>

# 🥔 MicroWakeWord Trainer – Tater Approved  

**✅ Verified working on RTX 3070 (8 GB VRAM)**  
A full-featured Dockerized environment for generating and training **microWakeWord** models with GPU acceleration.  
Now featuring a **Streamlit web UI** — no Jupyter notebook needed.

---

## 🚀 Quick Start  

### 1️⃣ Pull the Pre-Built Image

docker pull ghcr.io/tatertotterson/microwakeword:latest

---

### 2️⃣ Run the Container
```
docker run --rm -it \
  --gpus all \
  -p 8502:8502 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```
**What this does:**
- `--gpus all` → Enables full CUDA acceleration  
- `-p 8502:8502` → Exposes Streamlit web UI  
- `-v $(pwd):/data` → Mounts your working directory for model outputs  

---

### 3️⃣ Open the Trainer UI  

Visit [http://localhost:8502](http://localhost:8502) in your browser.  
You’ll see the **microWakeWord Trainer Dashboard**, powered by Streamlit.

---

## 🧰 Modes

### 🧱 Run Once (Setup)
Performs all one-time steps:
- Environment and GPU validation  
- Installs `microWakeWord` and dependencies  
- Prepares all background and augmentation datasets  

Use this first when you start from a fresh container or new system.

---

### 🎤 Generate (Train a New Wake Word)
End-to-end training pipeline that:
1. Generates TTS samples for your chosen wake word  
2. Builds augmentations and spectrogram features  
3. Trains the TensorFlow model (GPU accelerated)  
4. Exports `.tflite` and `.json` files ready for ESPHome or on-device inference  

You can customize:
- Wake word name  
- Number of generated samples  
- Piper voice model (English, German, French, or Dutch)

---

## 🧹 Clean Runs Made Easy  

Each training run automatically clears previous outputs:
training_parameters.yaml  
trained_models/  
generated_augmented_features/  
generated_samples/  

so you can retrain new wake words back-to-back without manually deleting data for fast repeat runs.

---

## 📦 Outputs  

After completion, download links for your model files will appear directly in the **web UI**:

- `<your_wakeword>.tflite` – the trained model  
- `<your_wakeword>.json` – metadata and configuration  

Both files are also saved in your `/data` directory, ready to drop into **ESPHome** or any **microWakeWord-compatible** inference engine.

---

## ⚙️ System Requirements  

| Component | Recommended |
|------------|--------------|
| GPU | NVIDIA RTX 3060 + |
| VRAM | ≥ 6 GB |
| RAM | ≥ 16 GB |
| Disk | ≥ 30 GB free |
| CUDA | 12.x (included in container) |

---

## 🙌 Credits  
Based on the brilliant work of [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord).  

---




