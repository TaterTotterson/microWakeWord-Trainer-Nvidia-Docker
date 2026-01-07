<div align="center">
  <img src="https://raw.githubusercontent.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
  <h1>microWakeWord Trainer Docker</h1>
</div>

# 🥔 MicroWakeWord Trainer – Tater Approved  

**✅ Tater Totterson tested & working on an NVIDIA RTX 3070 Laptop GPU (8 GB VRAM).**  
Easily train microWakeWord detection models with this pre-built Docker image and JupyterLab notebook.  

---

## 🚀 Quick Start  

Follow these steps to get up and running:  

### 1️⃣ Pull the Pre-Built Docker Image  

```bash
docker pull ghcr.io/tatertotterson/microwakeword:latest
```

---

### 2️⃣ Run the Docker Container  

```bash
docker run --rm -it \
    --gpus all \
    -p 8888:8888 \
    -v $(pwd):/data \
    ghcr.io/tatertotterson/microwakeword:latest
```

**What these flags do:**  
- `--gpus all` → Enables GPU acceleration  
- `-p 8888:8888` → Exposes JupyterLab on port 8888  
- `-v $(pwd):/data` → Saves your work in the current folder  

---

### 3️⃣ Open JupyterLab  

Visit [http://localhost:8888](http://localhost:8888) in your browser — the notebook UI will open.  

---

### 4️⃣ Set Your Wake Word  

At the **top of the notebook**, find this line:  

```bash
TARGET_WORD = "hey_tater"  # Change this to your desired wake word
```

Change `"hey_tater"` to your desired wake word (phonetic spellings often work best).  

---

### 5️⃣ Run the Notebook  

Run all cells in the notebook. This process will:  
- Generate wake word samples  
- Train a detection model  
- Output a quantized `.tflite` model ready for on-device use  

---

### 6️⃣ Retrieve the Trained Model & JSON  

When training finishes, download links for both the `.tflite` model and its `.json` manifest will be displayed in the last cell.  

---

## 🔄 Resetting to a Clean State  

If you need to start fresh:  

1. Delete the `data` folder that was mapped to your Docker container.  
2. Restart the container using the steps above.  
3. A fresh copy of the notebook will be placed into the `data` directory.  

---

## 🎤 Optional: Personal Voice Samples

In addition to synthetic TTS samples, the trainer can optionally use your own real voice recordings to significantly improve accuracy for your voice and environment.

### How it works
- If a folder named personal_samples/ exists and contains .wav files, the trainer will:
  - Automatically extract features from those recordings
  - Include them during training alongside the synthetic TTS data
  - Up-weight your personal samples during training for better real-world performance

No extra flags or configuration are required — it is detected automatically.

### How to use it
1. Create a folder in the repo root:
   mkdir personal_samples

2. Record yourself saying the wake word naturally and save the files as .wav:
   personal_samples/
     hey_tater_01.wav
     hey_tater_02.wav
     hey_tater_03.wav
     ...

3. Run the training script as normal:

If personal samples are found, you’ll see a message during training indicating they are being included.

### Recording tips
- 10–30 recordings is usually enough to see a noticeable improvement
- Vary distance, volume, and tone slightly
- Record in the same environment where the wake word will be used (room noise matters)
- Use 16-bit WAV files if possible (most recorders do this by default)

---

## 🙌 Credits  

This project builds upon the excellent work of [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord).  
Huge thanks to the original authors for their contributions to the open-source community!
