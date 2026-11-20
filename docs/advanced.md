# Advanced trainer guide

[Back to the README](../README.md)

Detailed recording, training, model-file, and development reference. Run shell
commands from the repository root unless a section says otherwise.

## Captured Audio Workflow

To collect samples from a sat, point its trainer feedback setting at this app. Tater Native satellites use their settings in Tater or the Home Assistant bridge. Audio uploads do not require the optional auto-training link. Older ESPHome satellites can still use their device entities.

For Tater Native satellites, enable trainer feedback in Tater:

- `Send Good Wakes To Trainer` toggles upload of confirmed wake-word triggers.
- `Send Close Misses To Trainer` toggles upload of near misses.
- `Trainer App URL` sets the trainer address, for example `http://trainer.local:8789` or `http://<trainer-ip>:8789`.

For older ESPHome firmware, the equivalent capture setup is exposed as device entities:

- `Capture Wake Audio` toggles upload of wake-word triggers.
- `Capture Close Misses` toggles upload of near misses.
- `Trainer App URL` sets the trainer address, for example `http://<trainer-ip>:8789`.

Satellites send raw captured audio to:

```text
/api/upload_captured_audio_raw
```

Keep the training app running and reachable at the `Trainer App URL` while capture is enabled. The sats upload clips live; if the app is stopped or the URL is wrong, captured audio will not be saved.

In **Captures**:

- play each clip from the inbox
- choose **Add positive** for good wake-word clips
- choose **Mark negative** for false triggers
- discard clips that should not be used

Approved clips move into:

```text
/data/personal_samples/
```

False wakes move into:

```text
/data/negative_samples/
```

Captured audio is boosted for easier playback in the UI, then kept in the correct training format.

## Samples

The `Samples` tab is the sample library.

- `Personal` samples are positive examples of the wake word.
- `Negative` samples are reviewed false wakes or hard negatives.
- Both can be played back and removed one at a time.
- Manual upload is available here as an optional seed path.

Accepted manual upload formats include:

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- OPUS
- WEBM

Uploads are validated or converted with `ffmpeg` into:

```text
16 kHz / mono / 16-bit PCM WAV
```

Starting a new session does not clear samples. Use the clear buttons in `Samples` if you want to remove saved personal or negative clips.

## Training Flow

1. On `Start`, choose `Create a wake word`, or open a saved word to review results and JSON links without starting a session.
2. For a new word, enter the phrase, language, English accent, and voice-generation source. `System preview` checks pronunciation with the system voice.
3. Add or review personal recordings and negatives. Upload, playback, trimming, VAD, restore, and bulk import remain available in the sample manager.
4. Select any combination of `MicroWakeWord`, `openWakeWord`, `NanoWakeWord · CNN`, and `NanoWakeWord · GRU`, plus optional phrases to reject. Recovery actions are under `Additional training actions`.
5. Review the configuration and click `Start training`. The `Training` page shows the current stage and logs; training continues while you browse.
6. Open the completed word for its measured validation results and JSON links. For an existing phrase, `Train a new version` reuses its saved data.

Personal samples are optional. Training can run with zero personal samples after confirmation, using generated TTS samples and the stock negative datasets.

### Saved words, cached datasets, and model history

Every phrase started in the Trainer is added to the **Saved wake words** library. Switching to another saved word moves that word's personal samples, reviewed negatives, generated TTS corpus, augmented features, boundary examples, and retained training workspace back into the active paths. The previous word is archived under `/data/wake_word_library/words/<safe_word>/assets/`, so selecting a different phrase no longer discards or overwrites its expensive data.

Each completed run also creates an immutable snapshot under `/data/wake_word_library/words/<safe_word>/runs/`. The **Wake Words** tab shows the run history, selected engines, recall/false-trigger metrics, thresholds, and quality rating so multiple attempts can be compared and open any saved version for its own bundle JSON and ESPHome JSON URLs. These version-specific links keep referencing that version’s model bytes. `Use v… as current` validates the saved artifacts and changes the current publication without deleting recordings, caches, or other versions. The stable published JSON URL follows the selected current version and remains on the committed snapshot while another run trains. Failed attempts remain in history and cannot replace the current publication. Publish through Tater separately when satellites should switch to the selected version.

After a word has at least one detector bundle, **Retrain boundary only** rebuilds and calibrates only the wake-end model. It preserves MWW, OWW, and NWW detector weights and reuses the saved word's cached corpus and features. From the CLI, use:

```bash
train_wake_word --data-dir /data --boundary-only "hey tater"
```

Boundary-only mode requires the retained detector training config from an earlier full MicroWakeWord run.

Reviewed negative samples are converted into `/data/work/reviewed_negative_features/` and inserted into the training YAML as a hard-negative feature set when present.

The first OWW or NWW run downloads its separate trainer stack and caches it for later runs. NanoWakeWord reuses the direct-generated positive TTS corpus under `/data/work/wake_word_samples`, then adds personal positives, reviewed negatives, and its own adversarial negatives and held-out validation clips. CNN is the lightweight default architecture; GRU can be selected alone or alongside CNN. The streaming gate automatically calibrates threshold and consecutive-hit patience, and the validated policy is stored with each model in the bundle. Each successful stage updates only its section of the canonical Tater bundle while removing obsolete NWW architectures.

On RTX 50-series / Blackwell GPUs, the Blackwell Docker image keeps sample generation and augmentation in the normal Python 3.12 trainer environment, then runs only the TensorFlow training/export stage in `/data/.venv-blackwell` with Python 3.13 and the Blackwell-native TensorFlow wheel.

Immediately before TensorFlow starts, the trainer adds every pip-installed NVIDIA runtime library from the selected training environment—including cuSolver, cuSPARSE, cuFFT, and nvJitLink—to the child process library path. A GPU preflight prints the number of TensorFlow devices it registered, so a visible NVIDIA GPU cannot fall back to CPU without an explicit warning in the training log.

## Auto Training

`Auto Training` is an opt-in sample-review and retraining loop. It is disabled until you enter the exact wake phrase and enable it.

For each new wake-trigger clip sent to the trainer:

1. The selected local STT engine transcribes the audio.
2. If the transcript contains the configured wake phrase, the clip stays in `Captured Audio` for manual review by default.
3. If speech was transcribed but the wake phrase is absent, the clip moves to `/data/negative_samples/` as an auto-reviewed hard negative.
4. Empty transcripts, VAD-blocked captures, and captures for another wake word stay out of the automatic negative path.

Two optional cleanup rules are available:

- `Delete confirmed good wakes` removes normal wake-trigger clips after STT confirms the configured phrase.
- `Promote confirmed close misses` checks close misses that passed VAD and moves them to the personal positive samples only when STT confirms the configured phrase.

A close miss with an empty transcript or without the configured phrase stays in `Captured Audio`; it is never turned into a negative automatically. Saving Auto Training settings also scans existing eligible captures. Enabling close-miss promotion reviews previous unreviewed close misses, while enabling cleanup removes previously confirmed good wakes without transcribing them a second time.

Auto Training exposes only an engine selector. Faster Whisper is the recommended default and uses CUDA with `float16` when CTranslate2 can see an NVIDIA GPU, with a CPU `int8` fallback. The trainer manages `small.en` for English and `small` for other languages. Parakeet ONNX uses the managed INT8 `nemo-parakeet-tdt-0.6b-v3` model with CUDA and CPU fallback. Downloaded STT models are cached in `/data/auto_train_models/`.

Scheduled training runs only after the configured number of new automatic negatives has accumulated. A successful run securely publishes the trained wake-word name and JSON URL to the linked Tater instance. Tater saves it as the global satellite wake word and pushes the updated setting to every connected satellite, so no satellite firmware change is required.

The `Trainer public URL` must be reachable from the satellites. With the documented `--network host` command, the trainer can normally use the LAN address from the browser request or host network. If you open the UI as `http://localhost:8789`, enter a value such as `http://192.168.1.50:8789`, or start the container with `REC_PUBLIC_BASE_URL` set to that value. When using Docker bridge networking, always set this URL to the published host address; a container bridge address is not satellite-reachable.

The default Tater URL, `http://127.0.0.1:8501`, assumes the documented host networking. Change it to a container-reachable Tater address if you use another Docker network. Click `Link Tater` and enter the short-lived code shown in Tater Voice Settings; the resulting trainer-specific link credential is stored in `/data/auto_train_config.json` with owner-only permissions.

The auto-training link is optional. Local review and training can be used without pairing; disable automatic activation if you are not linking a controller. For Home Assistant pairing and shared wake-profile behavior, see the [bridge guide](https://github.com/TaterTotterson/Tater-Home-Assistant-Satellites/blob/main/docs/user-guide.md#link-the-wake-word-trainer).

## Language Support

The language picker is built from OmniVoice's live catalog (currently more than 600 languages), with a bundled common-language fallback for offline startup. Languages covered by Qwen3-TTS and MOSS-TTS-Nano are automatically marked `Recommended`; OmniVoice-only languages are marked `Experimental` so lower-resource coverage is not presented as equal quality.

The selected code is sent directly to the supporting model. Model downloads happen only when a language is used, and the Hugging Face cache is persisted under `/data/.cache/huggingface`. The fetched language catalog is cached under `/data/.cache/omnivoice_languages.json`.

### TTS modes

- `Four-provider ensemble` is the default. It uses OmniVoice for every catalog language, adds Qwen3-TTS and MOSS-TTS-Nano where supported, and adds Piper when a compatible model exists.
- `Modern only` uses the multilingual providers without Piper.
- `Piper only` preserves the previous generator as an explicit legacy fallback.

Where a Piper voice is unavailable, the default route automatically continues with the modern providers.

Qwen, OmniVoice, and Piper now generate final corpus candidates directly instead of cloning a 128-profile bank. Qwen provides 18,750 balanced voice conditions before an instruction repeats, and Piper uses every speaker in its installed model. MOSS Nano is clone-only, so each MOSS take uses a different already-accepted direct take as its carrier rather than cycling a small bank.

Every generated file is normalized to `16 kHz / mono / 16-bit PCM WAV` and rejected if it contains static, broadband/high-frequency noise, silence, clipping, excessive duration/rambling, or an exact duplicate. A generation manifest records the planned and accepted provider counts and the applied safety limits.

## Dataset Behavior

The first training run downloads and prepares missing training assets into `/data`, including:

- isolated Python environments for each selected modern TTS engine
- selected TTS model weights and the direct generated corpus
- additional language-specific Piper voices only when hybrid or legacy Piper mode is selected
- negative datasets and background data
- the Python training environment
- generated samples and augmented feature caches

The three modern engines deliberately use separate environments under `/data/tts-envs/`; their required PyTorch and Transformers versions conflict with one another and with the trainer environment. Model weights can require many gigabytes, so allow extra disk space and time on the first run. After the assets are prepared, later runs reuse the local copies unless the mounted `/data` contents are deleted.

## Output Files

Successful runs produce timestamped training output folders such as:

```text
/data/output/<timestamp>-<wake_word>-<samples>-<steps>/<wake_word>.tflite
/data/output/<timestamp>-<wake_word>-<samples>-<steps>/<wake_word>.json
```

The trainer also syncs Tater-ready wake-word artifacts into:

```text
/data/trained_wake_words/<wake_word>.tflite
/data/trained_wake_words/<wake_word>.oww.onnx         # openWakeWord classifier
/data/trained_wake_words/<wake_word>.oww.verifier.pkl # when positive + negative clips exist
/data/trained_wake_words/<wake_word>.nww.cnn.onnx     # NanoWakeWord CNN classifier
/data/trained_wake_words/<wake_word>.nww.gru.onnx     # NanoWakeWord GRU classifier
/data/trained_wake_words/<wake_word>.boundary.tflite  # detector-independent wake-end model
/data/trained_wake_words/<wake_word>.wake-bundle.json
/data/trained_wake_words/<wake_word>.esphome.json     # MWW-only ESPHome manifest
```

Tater always receives `<wake_word>.wake-bundle.json`. Schema v4 embeds the complete metadata for every model that has been built, including hashes, thresholds, validation results, available NWW architectures, and one detector-independent `wake_boundary` TFLite artifact used to locate the acoustic end of the phrase. It does not point at a second MWW metadata JSON. ESPHome remains deliberately separate and MWW-only through `<wake_word>.esphome.json`.

The boundary artifact is trained and threshold-calibrated separately for each wake word. Wake-only clips remain positive while deterministic wake-plus-immediate-speech clips are hard negatives, forcing a causal score peak at the phrase endpoint. Packaging fails if that dedicated model was not exported; the normal wake detector is never substituted as boundary weights.

When MWW is selected, the bundle's `micro_wake_word` section keeps the standard microWakeWord fields:

```json
{
  "micro_wake_word": {
    "micro": {
      "probability_cutoff": 0.97,
      "sliding_window_size": 6
    }
  }
}
```

That same embedded section includes Tater Native metadata used by newer satellites and the Tater settings UI:

```json
{
  "micro_wake_word": {
    "model_format": "tflite_stream_state_internal_quant",
    "quantization": "int8",
    "sample_rate": 16000,
    "tater_native": {
      "format_version": 1,
      "wake_threshold": 0.97,
      "wake_sliding_window": 6,
      "close_miss_threshold": 0.80,
      "frontend": {
        "name": "tflm_microfrontend",
        "sample_rate": 16000,
        "feature_duration_ms": 30,
        "feature_step_ms": 10,
        "feature_size": 40
      }
    }
  }
}
```

Calibration metrics are included under `calibration` so false accepts/hour and recall can be surfaced in the UI.
Calibration evaluates thresholds from `0.95` through `1.00` with sliding windows of `5`, `6`, and `7`. Among candidates within 0.5 percentage points of the best recall, it prefers the lowest measured ambient false-accept rate. If calibration cannot complete, packaging uses the conservative `0.97` threshold and a window of `6`.

## Resetting Everything

Use **Data & storage** to remove individual rebuildable categories first. Back up recordings, saved versions, and settings before a complete reset.

If you want a clean slate, stop the container and remove the contents of the mounted `/data` directory.

That removes:

- personal samples
- negative samples
- captured inbox clips
- downloaded Piper voices
- modern TTS environments, model weights, and completed direct-generated corpora
- cached datasets
- training environments
- trained models
- Auto Training settings, state, transcripts, and cached Faster Whisper models

## UI development

The production bundle is committed under `static/ui`, so users do not need Node.js. Edit `frontend/src`, then build from the repository root:

```bash
cd frontend
npm install
npm run build
npm test
```

The build type-checks the Vue components and writes the offline bundle used by `trainer_server.py`. Tests require Node.js 22.6 or newer.

The UI uses `/api/events` (server-sent events), reconnects automatically, and falls back to polling. Returning to a browser tab catches up without resetting unsaved edits. Background library updates avoid recursive storage scans. If using a reverse proxy, allow unbuffered streaming responses on `/api/events`.

The UI server is `trainer_server.py`; its launcher is `run.sh`. Both launchers bound the shutdown wait for open live connections.

Before tagging a Docker release, update `VERSION`, rebuild the UI, and update `WHATS_NEW.md`. The tag must match `VERSION`; `.github/workflows/docker-publish.yml` publishes standard CUDA and Blackwell images plus the release notes.

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)
- [tensorflow-blackwell-python313](https://github.com/chivitiH/tensorflow-blackwell-python313) for the optional RTX 50-series / Blackwell image
