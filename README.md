# microWakeWord Trainer (CLI / RunPod)

Train **microWakeWord** detection models from the command line, optimized for **RunPod GPU pods**.

No web UI, no Jupyter notebooks — just SSH in, set up, and train.

---

## Quick Start

```bash
# 1. SSH into your RunPod pod
# 2. Run setup (one time — installs Python env + downloads ~50GB of datasets)
setup

# 3. Train a wake word
train_wake_word "hey jarvis"
```

That's it. Your trained model will be in `/data/output/`.

---

## RunPod Setup (Step by Step)

### 1. Create a RunPod Account

Go to [runpod.io](https://www.runpod.io/) and create an account.

### 2. Create a Network Volume

You need persistent storage for datasets and models (~100GB recommended).

1. Go to **Storage** in the RunPod dashboard
2. Click **Create Network Volume**
3. Set size to **100 GB** (or more)
4. Choose the same region you'll use for your pod
5. Name it something like `mww-data`

### 3. Create a GPU Pod

1. Go to **Pods** → **Deploy**
2. Choose a GPU (any NVIDIA GPU works — A40, RTX 4090, RTX 3090, etc.)
3. Under **Template**, select **Custom**
4. Set the **Docker image** to your built image (see [Building the Docker Image](#building-the-docker-image) below), or use:
   ```
   ghcr.io/bigpappy098/microwakeword-trainer:latest
   ```
5. Under **Volume**, attach your network volume and set the mount path to:
   ```
   /data
   ```
6. **(Optional)** Set environment variables for GitHub integration (see [GitHub Integration](#github-integration)):
   - `GITHUB_TOKEN` = your GitHub personal access token
   - `GITHUB_REPO` = `owner/repo` (e.g., `myuser/my-wakewords`)
7. Click **Deploy**

### 4. Connect via SSH

Once the pod is running:

1. Click **Connect** on your pod
2. Use the SSH command provided, e.g.:
   ```bash
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
   ```
3. You'll see a welcome message with available commands

### 5. Run Setup (First Time Only)

```bash
setup
```

This does two things:
1. **Creates the Python virtual environment** — installs TensorFlow, PyTorch, and all training dependencies into `/data/.venv`
2. **Downloads training datasets** — background noise, speech corpora, room impulse responses (~50GB)

This takes a while on the first run. Everything is cached in `/data`, so it persists across pod restarts.

### 6. Train a Wake Word

```bash
train_wake_word "hey jarvis"
```

The training pipeline:
1. **Generates** synthetic voice samples using TTS
2. **Augments** samples with background noise, room effects, pitch shifts, etc.
3. **Trains** a neural network (TensorFlow)
4. **Outputs** a quantized `.tflite` model + `.json` metadata
5. **Pushes to GitHub** (if configured)

#### Training Options

```bash
train_wake_word [options] <wake_word> [<wake_word_title>]

Options:
  --samples=<N>           Number of TTS samples to generate (default: 50000)
  --batch-size=<N>        Samples per generation batch (default: 100)
  --training-steps=<N>    Training iterations (default: 40000)
  --language=<lang>       TTS language: "en", "nl", etc. (default: en)
  --cleanup-work-dir      Delete intermediate files after training
```

Examples:
```bash
# Quick test run (small sample/step count)
train_wake_word --samples=1000 --training-steps=500 "hey jarvis"

# Full training with custom title
train_wake_word --samples=50000 --training-steps=40000 "hey jarvis" "Hey Jarvis"

# Dutch wake word
train_wake_word --language=nl "hallo computer"
```

---

## Output Files

After training, your model files are saved to:

```
/data/output/<timestamp>-<wake_word>-<samples>-<steps>/
  <wake_word>.tflite    # Quantized model for microcontrollers
  <wake_word>.json      # ESPHome-compatible metadata
  logs/                 # Training logs and TensorBoard data
```

The `.tflite` file is what you flash to your ESP32 or other device via ESPHome.

---

## Personal Voice Samples (Optional)

Recording your own voice improves accuracy for your specific voice. Since there's no microphone on RunPod, record `.wav` files on your local machine and upload them.

### Requirements
- 16kHz sample rate
- WAV format (PCM 16-bit)
- One wake word utterance per file

### Upload to RunPod

```bash
# From your local machine:
scp -P <port> speaker01_take*.wav root@<pod-ip>:/data/personal_samples/
```

### File naming convention

```
/data/personal_samples/
  speaker01_take01.wav
  speaker01_take02.wav
  speaker02_take01.wav
  ...
```

Personal samples are automatically detected and given **3x sampling weight** during training — no extra configuration needed.

---

## GitHub Integration

Automatically push trained model files to a GitHub repository after training.

### Setup

Set these environment variables (in RunPod pod settings or in your shell):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GITHUB_TOKEN` | Yes | — | GitHub Personal Access Token with `repo` scope |
| `GITHUB_REPO` | Yes | — | Target repo in `owner/repo` format |
| `GITHUB_BRANCH` | No | `main` | Branch to push to |
| `GITHUB_PATH` | No | `.` | Directory within the repo for model files |
| `GITHUB_COMMIT_MSG` | No | Auto-generated | Custom commit message |

### Create a GitHub Token

1. Go to [GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Give it a name like `mww-trainer`
4. Select the `repo` scope
5. Copy the token

### Set Environment Variables

In RunPod, add them under your pod's **Environment Variables**:
```
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO=yourusername/your-wakewords-repo
```

Or export them in your SSH session:
```bash
export GITHUB_TOKEN=ghp_your_token_here
export GITHUB_REPO=yourusername/your-wakewords-repo
```

After training completes, the `.tflite` and `.json` files are automatically committed and pushed. If the env vars aren't set, this step is silently skipped.

---

## Building the Docker Image

To build and push your own image:

```bash
git clone https://github.com/BigPappy098/microWakeWord-Trainer-Nvidia-Docker.git
cd microWakeWord-Trainer-Nvidia-Docker

docker build -t your-registry/mww-trainer:latest .
docker push your-registry/mww-trainer:latest
```

The image uses `nvidia/cuda:12.9.0-runtime-ubuntu24.04` as the base, so it works on any NVIDIA GPU.

---

## Other Useful Commands

Once SSH'd into the pod:

```bash
setup                    # Run full setup (venv + datasets)
setup_python_venv        # Set up just the Python environment
setup_training_datasets  # Download just the training datasets
cudainfo                 # Show GPU information
system_summary           # Show system stats (CPU, RAM, disk, GPU)
nvidia-smi               # NVIDIA GPU status
```

---

## Re-training and Multiple Wake Words

- Train **multiple wake words** back-to-back — no cleanup needed between runs
- Each run creates a **new timestamped output directory**
- Old models are preserved
- Intermediate work files in `/data/work/` are reused when possible

---

## Resetting Everything

To start completely fresh, delete the data volume contents:

```bash
rm -rf /data/*
```

This removes cached datasets, the Python venv, and all trained models. You'll need to run `setup` again.

---

## Storage Requirements

| Directory | Purpose | Size |
|---|---|---|
| `/data/.venv/` | Python environment | ~5 GB |
| `/data/training_datasets/` | Audio corpora | ~40 GB |
| `/data/tools/` | Git clones + TTS models | ~3 GB |
| `/data/work/` | Temporary training artifacts | ~10 GB |
| `/data/output/` | Trained models | ~10 MB per model |
| **Total** | | **~60 GB minimum** |

A 100 GB network volume is recommended to have comfortable headroom.

---

## Credits

Built on top of [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt.
