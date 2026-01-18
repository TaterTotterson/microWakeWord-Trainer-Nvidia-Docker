<div align="center">
  <h1>🎙️ microWakeWord Nvidia Trainer & Recorder</h1>
  <img width="1002" height="593" alt="Screenshot 2026-01-18 at 8 13 35 AM" src="https://github.com/user-attachments/assets/e1411d8a-8638-4df8-992b-09a46c6e5ddc" />
</div>

Train **microWakeWord** detection models using a simple **web-based recorder + trainer UI**, packaged in a Docker container.

No Jupyter notebooks required. No manual cell execution. Just record your voice (optional) and train.

---

## 🚀 Quick Start

### 1️⃣ Pull the Docker Image

```bash
docker pull ghcr.io/tatertotterson/microwakeword:latest
```

---

### 2️⃣ Run the Container

```bash
docker run --rm -it \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```

**What these flags do:**
- `--gpus all` → Enables GPU acceleration  
- `-p 8888:8888` → Exposes the Recorder + Trainer WebUI  
- `-v $(pwd):/data` → Persists all models, datasets, and cache  

---

### 3️⃣ Open the Recorder WebUI

Open your browser and go to:

👉 **http://localhost:8888**

You’ll see the **microWakeWord Recorder & Trainer UI**.

---

## 🎤 Recording Voice Samples (Optional)

Personal voice recordings are **optional**.

- You may **record your own voice** for better accuracy  
- Or simply **click “Train” without recording anything**

If no recordings are present, training will proceed using **synthetic TTS samples only**.

### Remote systems (important)
If you are running this on a **remote PC / server**, browser-based recording will not work unless:
- You use a **reverse proxy** (HTTPS + mic permissions), **or**
- You access the UI via **localhost** on the same machine

Training itself works fine remotely — only recording requires local microphone access.

---

## 🧠 Training Behavior (Important Notes)

### ⏬ First training run
The **first time you click Train**, the system will download **large training datasets** (background noise, speech corpora, etc.).

- This can take **several minutes**
- This happens **only once**
- Data is cached inside `/data`

You **will NOT need to download these again** unless you delete `/data`.

---

### 🔁 Re-training is safe and incremental

- You can train **multiple wake words** back-to-back
- You do **NOT** need to clear any folders between runs
- Old models are preserved in timestamped output directories
- All required cleanup and reuse logic is handled automatically

---

## 📦 Output Files

When training completes, you’ll get:
- `<wake_word>.tflite` – quantized streaming model  
- `<wake_word>.json` – ESPHome-compatible metadata  

Both are saved under:

```text
/data/output/
```

Each run is placed in its own timestamped folder.

---

## 🎤 Optional: Personal Voice Samples (Advanced)

If you record personal samples:
- They are automatically augmented
- They are **up-weighted during training**
- This significantly improves real-world accuracy

No configuration required — detection is automatic.

---

## 🔄 Resetting Everything (Optional)

If you want a **completely clean slate**:

Delete the /data folder

Then restart the container.

⚠️ This will:
- Remove cached datasets
- Require re-downloading training data
- Delete trained models

---

## 🙌 Credits

Built on top of the excellent  
**https://github.com/kahrendt/microWakeWord**

Huge thanks to the original authors ❤️
