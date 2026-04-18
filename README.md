<div align="center">
  <h1>microWakeWord NVIDIA Docker Trainer UI</h1>
  <img width="800" alt="Screenshot 2026-04-14 at 11 02 06 PM" src="https://github.com/user-attachments/assets/694f4cb7-e4d8-4e2b-80ec-b40fb41cbfff" />
</div>

Train custom microWakeWord models in Docker with:

- uploaded personal voice samples
- automatically generated Piper TTS samples
- a browser-based trainer UI
- live training logs in a popup console

This project no longer records audio in the browser. The UI is now upload-first: users add their own audio files, the app validates or converts them, and training runs from the same page.

---

## Docker Image

```bash
docker pull ghcr.io/tatertotterson/microwakeword:latest
```

---

## Run The Container

```bash
docker run -d \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```

What these flags do:

- `--gpus all` enables GPU acceleration
- `-p 8888:8888` exposes the trainer UI
- `-v $(pwd):/data` persists models, downloaded voices, datasets, and personal samples

Then open:

```text
http://localhost:8888
```

---

## What The UI Does

- Start a wake word session
- Test TTS pronunciation
- Upload one or many personal samples
- Normalize uploads to `16 kHz / mono / 16-bit PCM WAV`
- Train with or without personal samples
- Show a popup console with live progress and logs

Personal samples are optional. If none are uploaded, the trainer can still proceed with TTS-only data after confirmation.

---

## Personal Samples

Accepted upload formats include:

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- OPUS
- WEBM

The backend validates or converts uploads with `ffmpeg` and stores the normalized files in:

```text
/data/personal_samples/
```

Notes:

- starting a new session does not clear personal samples
- use the `Clear personal samples` button if you want to wipe them
- any uploaded personal samples are automatically included in training

---

## Language Support

The language selector is dynamic.

- `en` is always available
- non-English languages are populated from Piper voice metadata
- when you train with a non-English language, the backend downloads all Piper ONNX voices for that selected language only
- it does not pre-download every language
- already-downloaded voices are reused on later runs

English stays on its existing dedicated generator model path. Non-English languages use the selected language's ONNX Piper voices.

If the Piper catalog is unavailable, already-installed local voices can still be used.

---

## Training Behavior

1. Enter the wake word
2. Optionally test pronunciation
3. Optionally upload personal samples
4. Click `Start training`
5. Watch the popup console for:
   - selected-language voice downloads when needed
   - sample generation progress
   - dataset setup
   - training progress and completion

The `Open console` button lets you reopen the log window after closing it.

---

## First Run Notes

The first real training run may download large training assets into `/data`, such as:

- Piper voices for the selected language
- training datasets and background data
- Python training environment dependencies

These are reused later unless you delete `/data`.

---

## Output Files

Successful runs produce:

```text
/data/output/<wake_word>.tflite
/data/output/<wake_word>.json
```

If those files already exist, the trainer creates timestamped backups before replacing them.

---

## Resetting Everything

If you want a clean slate, stop the container and remove the contents of your mounted `/data` directory.

That will remove:

- personal samples
- downloaded Piper voices
- cached datasets
- training environments
- trained models

---

## Notes

- browser microphone recording has been removed
- personal samples are optional
- the server module is now `trainer_server.py`
- the launcher script is now `run.sh`

---

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
