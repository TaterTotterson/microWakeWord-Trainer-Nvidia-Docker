#!/usr/bin/env python3
"""Generate a normalized wake-word corpus from the modern TTS ensemble.

Heavy model dependencies live in per-engine virtual environments.  This
orchestrator itself is standard-library-only so it can safely run from the
trainer environment without changing TensorFlow, PyTorch, or Transformers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import wave
from array import array
from collections import Counter
from itertools import product
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tts_config import (  # noqa: E402
    DEFAULT_TTS_MODE,
    ENGINE_MOSS,
    ENGINE_OMNIVOICE,
    ENGINE_PIPER,
    ENGINE_QWEN3,
    QWEN_LANGUAGE_NAMES,
    distribute_samples,
    engines_for_language,
    language_for_engine,
    normalize_tts_mode,
)


GENERATOR_VERSION = "modern-tts-v16-four-provider-direct-corpus-safe-limits"
VOICE_BANK_VERSION = "modern-tts-voice-bank-v1-native-random-qualified-single-utterance"
COMPATIBLE_VOICE_BANK_VERSIONS = {
    VOICE_BANK_VERSION,
    "modern-tts-v11-native-random-qualified-reference-only-semantic-single-utterance-voices",
}
OMNIVOICE_MODEL = "k2-fsa/OmniVoice"
SPEEDS = (0.85, 0.95, 1.0, 1.05, 1.15)
VOICE_PROFILE_STRIDE = 293
OMNIVOICE_PROMPT_RETRY_ROUNDS = 4
OMNIVOICE_CORPUS_RETRY_ROUNDS = 3
MOSS_CORPUS_RETRY_ROUNDS = 3
VOICE_BANK_REPLACEMENT_ROUNDS = 6
OMNIVOICE_REPLACEMENT_FACTOR = 2.0
OMNIVOICE_POSITION_TEMPERATURE = 5.0
OMNIVOICE_CLASS_TEMPERATURE = 0.0
DIRECT_CANDIDATE_FACTORS = {
    ENGINE_OMNIVOICE: 1.50,
    ENGINE_QWEN3: 1.08,
    ENGINE_MOSS: 1.25,
    ENGINE_PIPER: 1.05,
}
NORMALIZATION_TIMEOUT_SECONDS = 30.0
NORMALIZATION_PROGRESS_INTERVAL = 100

CARRIER_PROMPT_TEMPLATES = {
    "ar": "بصوت هادئ وطبيعي أقول {phrase} بوضوح، ثم أواصل الحديث بإيقاع ثابت.",
    "cs": "Klidným a přirozeným hlasem zřetelně řeknu {phrase} a potom pokračuji rovnoměrným tempem.",
    "da": "Med en rolig og naturlig stemme siger jeg {phrase} tydeligt og fortsætter derefter i et jævnt tempo.",
    "de": "Mit ruhiger und natürlicher Stimme sage ich deutlich {phrase} und spreche danach in gleichmäßigem Tempo weiter.",
    "el": "Με ήρεμη και φυσική φωνή λέω καθαρά {phrase} και μετά συνεχίζω να μιλάω με σταθερό ρυθμό.",
    "en": "In a calm and natural voice, I say {phrase} clearly, then continue speaking at an even pace.",
    "es": "Con una voz tranquila y natural, digo {phrase} con claridad y después sigo hablando a un ritmo constante.",
    "fa": "با صدایی آرام و طبیعی، عبارت {phrase} را واضح می‌گویم و سپس با ریتمی یکنواخت ادامه می‌دهم.",
    "fr": "D’une voix calme et naturelle, je dis clairement {phrase}, puis je continue à parler à un rythme régulier.",
    "hu": "Nyugodt és természetes hangon tisztán kimondom, hogy {phrase}, majd egyenletes tempóban folytatom.",
    "it": "Con una voce calma e naturale, dico chiaramente {phrase} e poi continuo a parlare a un ritmo regolare.",
    "ja": "落ち着いた自然な声で {phrase} とはっきり言い、そのまま一定の速さで話し続けます。",
    "ko": "차분하고 자연스러운 목소리로 {phrase}라고 또렷하게 말한 뒤 일정한 속도로 계속 말합니다.",
    "pl": "Spokojnym i naturalnym głosem wyraźnie mówię {phrase}, a potem kontynuuję w równym tempie.",
    "pt": "Com uma voz calma e natural, digo {phrase} com clareza e depois continuo falando em um ritmo constante.",
    "ru": "Спокойным и естественным голосом я чётко произношу {phrase}, а затем продолжаю говорить в ровном темпе.",
    "sv": "Med en lugn och naturlig röst säger jag {phrase} tydligt och fortsätter sedan i en jämn takt.",
    "tr": "Sakin ve doğal bir sesle {phrase} ifadesini açıkça söylüyor, ardından düzenli bir hızda konuşmaya devam ediyorum.",
    "zh": "我会用平静自然的声音清楚地说 {phrase}，然后保持均匀的语速继续说话。",
}


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    log("→ " + " ".join(command))
    # Some upstream CLIs terminate their entire process group after a fatal
    # worker error.  Give each model command its own group so that behavior
    # cannot kill the trainer/orchestrator process.
    subprocess.run(command, check=True, env=env, start_new_session=True)


def run_with_batch_retry(
    command: list[str],
    batch_flag: str,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Retry a failed batched model command with a single-item batch."""

    batch_index = command.index(batch_flag) + 1
    preferred = int(command[batch_index])
    try:
        run(command, env=env)
    except subprocess.CalledProcessError:
        if preferred <= 1:
            raise
        log(f"⚠️ Batch size {preferred} failed; retrying with batch size 1")
        retry_command = list(command)
        retry_command[batch_index] = "1"
        run(retry_command, env=env)


def run_normalization_ffmpeg(command: list[str], timeout: float) -> int | None:
    """Run one conversion without allowing a stuck file read to block training."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Do not use subprocess.run(timeout=...) here. On POSIX it performs an
        # unbounded wait after killing the child, which can still freeze the
        # trainer when a process is stuck in filesystem I/O.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            # The process may remain in uninterruptible I/O until the kernel
            # releases it. The next candidate can still be processed safely.
            pass
        return None


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def phrase_key(phrase: str) -> str:
    return hashlib.sha256(phrase.encode("utf-8")).hexdigest()[:16]


def reference_text(phrase: str) -> str:
    clean = phrase.strip().rstrip(".!?")
    return clean + "."


def duration_bounds(phrase: str, language: str) -> tuple[float, float, float]:
    """Return a natural target and strict min/max for one wake utterance."""

    clean = phrase.replace("_", " ").strip()
    words = [word for word in clean.split() if word]
    if len(words) > 1:
        units = len(words)
    else:
        units = max(1.0, len(clean) / (2.5 if language in {"zh", "ja", "ko"} else 5.0))
    target = min(3.2, max(0.9, 0.55 + (0.48 * units)))
    minimum = max(0.25, min(0.65, target * 0.35))
    maximum = min(5.0, max(1.8, target * 1.8))
    return minimum, target, maximum


def stable_prompt_text(phrase: str, language: str = "en") -> str:
    clean = phrase.strip().rstrip(".!?")
    template = CARRIER_PROMPT_TEMPLATES.get(language.strip().lower().split("_", 1)[0])
    if template:
        return template.format(phrase=clean)
    # Experimental OmniVoice languages may not have a trustworthy carrier
    # translation yet. A single clean utterance is safer than repeating the
    # wake phrase, which can make a cloning prompt collapse into humming.
    return clean + "."


def qwen_descriptions(language_name: str, count: int) -> list[str]:
    genders = ("female", "male")
    ages = ("child", "teenager", "young adult", "middle-aged adult", "elderly adult")
    pitches = ("low pitch", "medium pitch", "high pitch")
    deliveries = (
        "calm neutral delivery",
        "bright energetic delivery",
        "soft careful delivery",
        "confident resonant delivery",
        "casual conversational delivery",
    )
    textures = ("clear", "warm", "slightly breathy", "crisp", "gently rough")
    paces = ("slow", "measured", "natural", "brisk", "quick")
    weights = ("light", "balanced", "compact", "full-bodied", "resonant")
    combinations = list(product(genders, ages, pitches, deliveries, textures, paces, weights))
    descriptions = []
    # Walking the Cartesian product sequentially clusters the leading traits
    # (the first 375 combinations are all female).  A coprime stride retains a
    # deterministic, non-repeating order while balancing every trait early.
    for index in range(count):
        combination_index = (index * VOICE_PROFILE_STRIDE) % len(combinations)
        gender, age, pitch, delivery, texture, pace, weight = combinations[combination_index]
        descriptions.append(
            f"A distinct {age} {gender} speaker with a {texture} timbre, "
            f"{pitch}, {weight} vocal weight, and {delivery}, speaking native "
            f"{language_name} at a {pace} pace. Say only the supplied text once."
        )
    return descriptions


def omnivoice_stability_args() -> list[str]:
    # A zero temperature can deterministically lock a difficult voice condition
    # into decoder collapse. Retain upstream sampling and qualify each native
    # carrier through a short reference clone before it can enter the bank.
    return [
        "--position_temperature",
        str(OMNIVOICE_POSITION_TEMPERATURE),
        "--class_temperature",
        str(OMNIVOICE_CLASS_TEMPERATURE),
    ]


def read_pcm_metrics(path: Path) -> tuple[float, float, float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            raw = wav_file.readframes(frames)
    except Exception:
        return (0.0, 0.0, 1.0)
    if rate <= 0 or width != 2 or channels <= 0 or not raw:
        return (0.0, 0.0, 1.0)
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return (0.0, 0.0, 1.0)
    peak = max(abs(value) for value in samples) / 32767.0
    rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / 32767.0
    clipped = sum(1 for value in samples if abs(value) >= 32760) / len(samples)
    return (frames / rate, rms, clipped if peak > 0 else 1.0)


def valid_reference(path: Path) -> bool:
    duration, rms, clipped = read_pcm_metrics(path)
    return 0.2 <= duration <= 5.0 and rms >= 0.003 and clipped <= 0.08


def valid_prompt_reference(path: Path) -> bool:
    duration, rms, clipped = read_pcm_metrics(path)
    return 0.5 <= duration <= 10.0 and rms >= 0.003 and clipped <= 0.08


def valid_sample(path: Path) -> bool:
    duration, rms, clipped = read_pcm_metrics(path)
    return 0.12 <= duration <= 5.0 and rms >= 0.002 and clipped <= 0.08


class Generator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.spoken_phrase = args.phrase.replace("_", " ").strip()
        self.data_dir = args.data_dir.resolve()
        self.output_dir = args.output_dir.resolve()
        if self.output_dir == Path(self.output_dir.anchor) or self.output_dir == self.data_dir:
            raise ValueError(f"Refusing unsafe TTS output directory: {self.output_dir}")
        self.tools_dir = self.data_dir / "tools"
        self.tts_envs = self.data_dir / "tts-envs"
        self.hf_home = self.data_dir / ".cache" / "huggingface"
        self.work_root = self.output_dir.parent
        self.build_dir = self.work_root / ".wake_word_samples.build"
        self.raw_dir = self.build_dir / "raw"
        self.final_dir = self.build_dir / "final"
        self.voice_bank_dir = (
            self.data_dir
            / "voice-bank"
            / args.language
            / phrase_key(self.spoken_phrase)
        )
        self.reference_text = reference_text(self.spoken_phrase)
        self.stable_prompt_text = stable_prompt_text(self.spoken_phrase, args.language)
        self.omnivoice_language = language_for_engine(ENGINE_OMNIVOICE, args.language)
        self.env = dict(os.environ)
        self.env["HF_HOME"] = str(self.hf_home)
        self.env["HUGGINGFACE_HUB_CACHE"] = str(self.hf_home / "hub")
        self.speed_by_path: dict[Path, float] = {}
        self.actual_counts: dict[str, int] = {}
        self.reference_qa_batch = 0
        self.accepted_hashes: set[str] = set()
        self.direct_attempt = Counter()
        self.minimum_duration, self.target_duration, self.maximum_duration = duration_bounds(
            self.spoken_phrase, self.args.language
        )

    def piper_models(self) -> list[Path]:
        root = self.tools_dir / "piper-sample-generator"
        if self.args.language == "en":
            model = root / "models" / "en_US-libritts_r-medium.pt"
            return [model] if model.is_file() else []
        return sorted((root / "voices").glob(f"{self.args.language}_*.onnx"))

    def piper_available(self) -> bool:
        return bool(self.piper_models())

    def engines(self) -> list[str]:
        return engines_for_language(
            self.args.language,
            self.args.tts_mode,
            piper_available=self.piper_available(),
        )

    def signature(self) -> dict:
        engines = self.engines()
        return {
            "generator_version": GENERATOR_VERSION,
            "phrase": self.args.phrase,
            "language": self.args.language,
            "tts_mode": self.args.tts_mode,
            "samples": self.args.samples,
            "engines": engines,
            "models": {
                "omnivoice": OMNIVOICE_MODEL,
                "qwen_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "moss": "OpenMOSS-Team/MOSS-TTS-Nano",
                "piper": [str(path) for path in self.piper_models()],
            },
            "corpus_strategy": "direct_unique_candidates_with_provider_safety_gates",
            "duration_seconds": [self.minimum_duration, self.maximum_duration],
        }

    def cache_hit(self) -> bool:
        manifest_path = self.output_dir / ".generation_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            manifest.get("signature") == self.signature()
            and len(list(self.output_dir.glob("*.wav"))) == self.args.samples
        )

    def ensure_environment(self, engine: str) -> Path:
        if engine == ENGINE_PIPER:
            return self.data_dir / ".venv" / "bin" / "python"
        run(
            [
                str(ROOT_DIR / "cli" / "setup_modern_tts_envs"),
                f"--engine={engine}",
                f"--data-dir={self.data_dir}",
            ],
            env=self.env,
        )
        python = self.tts_envs / engine / "bin" / "python"
        if not python.is_file():
            raise RuntimeError(f"Missing {engine} Python environment: {python}")
        return python

    def _generate_omni_bank(self, count: int, start: int, destination: Path) -> list[dict]:
        if count <= 0:
            return []
        self.ensure_environment(ENGINE_OMNIVOICE)
        prompt_dir = destination / ".omnivoice-prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        # Generate unconstrained OmniVoice carriers natively. Explicit multi-
        # attribute voice instructions caused substantially more decoder
        # collapse in testing. The short clone below is the qualification
        # probe; only probes that pass speech/semantic QA enter the bank.
        prompt_entries = [
            {
                "id": f"omni_prompt_{start + index:04d}",
                "text": self.stable_prompt_text,
                "language_id": self.omnivoice_language,
            }
            for index in range(count)
        ]
        prompt_input = destination / f"omni_prompt_{start:04d}.jsonl"
        write_jsonl(prompt_input, prompt_entries)
        prompt_input_flag = "--test_list"
        prompt_batch_flag = "--batch_size"
        prompt_command = [
            str(self.tts_envs / ENGINE_OMNIVOICE / "bin" / "omnivoice-infer-batch"),
            "--model",
            OMNIVOICE_MODEL,
            prompt_input_flag,
            str(prompt_input),
            "--res_dir",
            str(prompt_dir),
            prompt_batch_flag,
            str(max(1, min(self.args.batch_size, 8))),
            "--lang_id",
            self.omnivoice_language,
        ] + omnivoice_stability_args()
        try:
            run_with_batch_retry(prompt_command, prompt_batch_flag, env=self.env)
        except subprocess.CalledProcessError as error:
            log(f"⚠️ Voice seed batch exited early; checking individual outputs: {error}")

        def missing_prompts() -> list[dict]:
            return [
                entry
                for entry in prompt_entries
                if not valid_prompt_reference(prompt_dir / f"{entry['id']}.wav")
            ]

        for retry_round in range(1, OMNIVOICE_PROMPT_RETRY_ROUNDS + 1):
            missing = missing_prompts()
            if not missing:
                break
            for entry in missing:
                (prompt_dir / f"{entry['id']}.wav").unlink(missing_ok=True)
            log(
                f"→ Voice seed repair round {retry_round}: retrying "
                f"{len(missing)} missing or invalid prompt(s) individually"
            )
            retry_input = destination / f"omni_prompt_{start:04d}.retry-{retry_round}.jsonl"
            write_jsonl(retry_input, missing)
            retry_command = list(prompt_command)
            retry_command[retry_command.index(prompt_input_flag) + 1] = str(retry_input)
            retry_command[retry_command.index(prompt_batch_flag) + 1] = "1"
            try:
                run(retry_command, env=self.env)
            except subprocess.CalledProcessError as error:
                log(f"⚠️ Voice seed repair round {retry_round} exited early: {error}")

        unresolved = missing_prompts()
        if unresolved:
            log(
                f"⚠️ OmniVoice could not create {len(unresolved)}/{count} stable seed "
                "prompt(s); replacement rounds will retry those profiles."
            )

        entries = []
        for index, prompt_entry in enumerate(prompt_entries):
            prompt_path = prompt_dir / f"{prompt_entry['id']}.wav"
            if not valid_prompt_reference(prompt_path):
                continue
            entries.append(
                {
                    "id": f"omni_ref_{start + index:04d}",
                    "text": self.reference_text,
                    "language_id": self.omnivoice_language,
                    "ref_audio": str(prompt_path),
                    "ref_text": self.stable_prompt_text,
                    # OmniVoice's batch API documents voice instructions for
                    # voice-design mode when no reference audio is present.
                    # The Qwen carrier already defines this voice, so cloning
                    # is deliberately reference-only to avoid conflicting
                    # conditioning on very short wake-word utterances.
                    "voice_description": "automatic random voice",
                    "omnivoice_prompt_path": str(prompt_path),
                    "omnivoice_prompt_text": self.stable_prompt_text,
                }
            )
        clone_input = destination / f"omni_clone_{start:04d}.jsonl"
        write_jsonl(clone_input, entries)
        clone_command = [
            str(self.tts_envs / ENGINE_OMNIVOICE / "bin" / "omnivoice-infer-batch"),
            "--model",
            OMNIVOICE_MODEL,
            "--test_list",
            str(clone_input),
            "--res_dir",
            str(destination),
            "--batch_size",
            str(max(1, min(self.args.batch_size, 8))),
            "--lang_id",
            self.omnivoice_language,
        ] + omnivoice_stability_args()
        if not entries:
            return []
        run_with_batch_retry(clone_command, "--batch_size", env=self.env)
        return entries

    def _generate_qwen_bank(self, count: int, destination: Path, start: int = 0) -> list[dict]:
        if count <= 0:
            return []
        python = self.ensure_environment(ENGINE_QWEN3)
        language_name = QWEN_LANGUAGE_NAMES[self.args.language]
        descriptions = qwen_descriptions(language_name, start + count)[start:]
        entries = [
            {
                "id": f"qwen_ref_{start + index:04d}",
                "text": self.reference_text,
                "language_name": language_name,
                "instruct": descriptions[index],
                "seed": 11000 + start + index,
            }
            for index in range(count)
        ]
        input_path = destination / "qwen_bank.jsonl"
        write_jsonl(input_path, entries)
        run_with_batch_retry(
            [
                str(python),
                str(ROOT_DIR / "cli" / "tts_qwen_worker.py"),
                "--mode",
                "bank",
                "--input-jsonl",
                str(input_path),
                "--output-dir",
                str(destination),
                "--batch-size",
                str(max(1, min(self.args.batch_size, 4))),
            ],
            "--batch-size",
            env=self.env,
        )
        return entries

    def _reference_qa_python(self) -> Path:
        candidates = []
        configured = os.environ.get("REC_VENV_DIR")
        if configured:
            candidates.append(Path(configured) / "bin" / "python")
        candidates.extend(
            (
                self.data_dir / ".recorder-venv" / "bin" / "python",
                self.data_dir.parent.parent / "recorder-venv" / "bin" / "python",
            )
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "Reference QA requires the trainer recorder environment with "
            "Faster Whisper and Silero VAD installed."
        )

    def _validate_generated_references(
        self,
        generated: list[tuple[dict, str]],
        destination: Path,
    ) -> list[dict]:
        candidates = []
        generated_by_id = {}
        acoustically_rejected = []
        for entry, source in generated:
            path = destination / f"{entry['id']}.wav"
            if not path.is_file() or not valid_reference(path):
                acoustically_rejected.append(path.name)
                continue
            candidates.append({"id": entry["id"], "path": str(path)})
            generated_by_id[entry["id"]] = (entry, source, path)
        if acoustically_rejected:
            examples = ", ".join(acoustically_rejected[:5])
            suffix = "" if len(acoustically_rejected) <= 5 else ", …"
            log(
                f"⚠️ Acoustic QA rejected {len(acoustically_rejected)} reference(s): "
                f"{examples}{suffix}"
            )
        if not candidates:
            return []

        self.reference_qa_batch += 1
        qa_input = destination / f"reference_qa_{self.reference_qa_batch:02d}.jsonl"
        qa_output = destination / f"reference_qa_{self.reference_qa_batch:02d}.results.jsonl"
        write_jsonl(qa_input, candidates)
        run(
            [
                str(self._reference_qa_python()),
                str(ROOT_DIR / "cli" / "tts_reference_qa.py"),
                "--input-jsonl",
                str(qa_input),
                "--output-jsonl",
                str(qa_output),
                "--phrase",
                self.spoken_phrase,
                "--language",
                self.args.language,
                "--download-root",
                str(self.data_dir / "auto_train_models"),
            ],
            env=self.env,
        )
        qa_results = {
            result["id"]: result
            for line in qa_output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for result in (json.loads(line),)
        }

        voices = []
        rejected_reasons: Counter[str] = Counter()
        rejected_examples: dict[str, list[str]] = {}
        for candidate in candidates:
            entry, source, path = generated_by_id[candidate["id"]]
            qa = qa_results.get(entry["id"])
            if not qa or not qa.get("accepted"):
                reason = (qa or {}).get("reason", "missing_qa_result")
                transcript = (qa or {}).get("transcript", "")
                rejected_reasons[reason] += 1
                examples = rejected_examples.setdefault(reason, [])
                if len(examples) < 3:
                    examples.append(f"{path.name}={transcript!r}")
                continue
            voice = {
                "id": entry["id"],
                "source": source,
                "path": str(path),
                "ref_text": self.reference_text,
                "language": self.args.language,
                "instruct": entry.get(
                    "voice_description",
                    entry.get("instruct", "automatic random voice"),
                ),
                "qa": qa,
            }
            if entry.get("omnivoice_prompt_path"):
                voice["omnivoice_prompt_path"] = entry["omnivoice_prompt_path"]
                voice["omnivoice_prompt_text"] = entry["omnivoice_prompt_text"]
            voices.append(voice)
        if rejected_reasons:
            summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(rejected_reasons.items())
            )
            examples = "; ".join(
                f"{reason}: {', '.join(items)}"
                for reason, items in sorted(rejected_examples.items())
            )
            log(f"⚠️ Reference QA rejected {sum(rejected_reasons.values())}: {summary}")
            if examples:
                log(f"   Examples: {examples}")
        return voices

    def ensure_voice_bank(self) -> list[dict]:
        qwen_target = self.args.voice_count // 2 if self.args.language in QWEN_LANGUAGE_NAMES else 0
        omni_target = self.args.voice_count - qwen_target
        source_targets = Counter(
            {
                ENGINE_OMNIVOICE: omni_target,
                **({ENGINE_QWEN3: qwen_target} if qwen_target else {}),
            }
        )
        manifest_path = self.voice_bank_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                voices = manifest.get("voices") or []
                source_counts = Counter(item.get("source") for item in voices)
                if (
                    manifest.get("version") in COMPATIBLE_VOICE_BANK_VERSIONS
                    and manifest.get("reference_text") == self.reference_text
                    and manifest.get("source_targets") == dict(source_targets)
                    and len(voices) >= self.args.voice_count
                    and all(source_counts.get(source, 0) >= target for source, target in source_targets.items())
                    and all(
                        Path(item["path"]).is_file()
                        and (
                            not item.get("omnivoice_prompt_path")
                            or Path(item["omnivoice_prompt_path"]).is_file()
                        )
                        for item in voices
                    )
                ):
                    log(f"✅ Reusing {len(voices)} cached voice profiles for {self.args.language}")
                    return voices[: self.args.voice_count]
            except Exception:
                pass

        building = self.voice_bank_dir.with_name(self.voice_bank_dir.name + ".building")
        shutil.rmtree(building, ignore_errors=True)
        building.mkdir(parents=True, exist_ok=True)
        generated: list[tuple[dict, str]] = []

        try:
            for entry in self._generate_omni_bank(omni_target, 0, building):
                generated.append((entry, ENGINE_OMNIVOICE))
        except Exception as error:
            log(f"⚠️ OmniVoice bank pass failed; its replacement rounds will retry: {error}")

        if qwen_target:
            try:
                for entry in self._generate_qwen_bank(qwen_target, building):
                    generated.append((entry, ENGINE_QWEN3))
            except Exception as error:
                log(f"⚠️ Qwen voice-design pass failed; its replacement rounds will retry: {error}")

        targets = source_targets
        voices = self._validate_generated_references(generated, building)
        next_index = {ENGINE_OMNIVOICE: self.args.voice_count * 2, ENGINE_QWEN3: self.args.voice_count * 2}
        for replacement_round in range(1, VOICE_BANK_REPLACEMENT_ROUNDS + 1):
            accepted_counts = Counter(voice["source"] for voice in voices)
            deficits = {
                source: target - accepted_counts.get(source, 0)
                for source, target in targets.items()
                if target > accepted_counts.get(source, 0)
            }
            if not deficits:
                break
            replacements: list[tuple[dict, str]] = []
            for source, missing in deficits.items():
                requested = max(
                    missing,
                    math.ceil(
                        missing
                        * (OMNIVOICE_REPLACEMENT_FACTOR if source == ENGINE_OMNIVOICE else 1.1)
                    ),
                )
                start = next_index[source]
                next_index[source] += requested
                log(
                    f"→ QA replacement round {replacement_round}: generating "
                    f"{requested} {source} candidate(s) for {missing} missing profile(s)"
                )
                if source == ENGINE_QWEN3:
                    try:
                        entries = self._generate_qwen_bank(requested, building, start)
                    except Exception as error:
                        log(f"⚠️ Qwen replacement round {replacement_round} failed: {error}")
                        continue
                else:
                    try:
                        entries = self._generate_omni_bank(requested, start, building)
                    except Exception as error:
                        log(f"⚠️ OmniVoice replacement round {replacement_round} failed: {error}")
                        continue
                replacements.extend((entry, source) for entry in entries)
            accepted_replacements = self._validate_generated_references(replacements, building)
            for voice in accepted_replacements:
                source = voice["source"]
                if accepted_counts.get(source, 0) < targets[source]:
                    voices.append(voice)
                    accepted_counts[source] = accepted_counts.get(source, 0) + 1

        accepted_counts = Counter(voice["source"] for voice in voices)
        source_deficits = {
            source: target - accepted_counts.get(source, 0)
            for source, target in targets.items()
            if accepted_counts.get(source, 0) < target
        }
        if source_deficits:
            raise RuntimeError(
                f"Voice bank did not reach its validated source mix: {source_deficits}."
            )

        self.voice_bank_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.voice_bank_dir, ignore_errors=True)
        shutil.move(str(building), str(self.voice_bank_dir))
        for voice in voices:
            for key in ("path", "omnivoice_prompt_path"):
                if voice.get(key):
                    relative_path = Path(voice[key]).relative_to(building)
                    voice[key] = str(self.voice_bank_dir / relative_path)
        manifest = {
            "version": VOICE_BANK_VERSION,
            "language": self.args.language,
            "reference_text": self.reference_text,
            "source_targets": dict(source_targets),
            "voices": voices,
        }
        (self.voice_bank_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log(f"✅ Created {len(voices)} reusable voice profiles")
        return voices[: self.args.voice_count]

    def make_entries(
        self,
        engine: str,
        count: int,
        voices: list[dict],
        destination: Path,
        prefix: str = "",
    ) -> list[dict]:
        rng = random.Random(24051984 + count + sum(ord(ch) for ch in engine + prefix))
        entries = []
        for index in range(count):
            voice_pool = voices
            if engine == ENGINE_OMNIVOICE:
                stable_omni_voices = [voice for voice in voices if voice.get("omnivoice_prompt_path")]
                if not stable_omni_voices:
                    raise RuntimeError(
                        "OmniVoice generation requires a validated long-form seed prompt."
                    )
                voice_pool = stable_omni_voices
            voice = voice_pool[index % len(voice_pool)]
            item_id = f"{engine}_{prefix}{index:06d}"
            # MOSS short utterances are fragile under post-generation tempo
            # changes; preserve the exact audio that passed semantic QA.
            speed = 1.0 if engine == ENGINE_MOSS else SPEEDS[index % len(SPEEDS)]
            entry = {
                "id": item_id,
                "text": self.spoken_phrase,
                "ref_audio": voice.get("omnivoice_prompt_path", voice["path"])
                if engine == ENGINE_OMNIVOICE
                else voice["path"],
                "ref_text": voice.get("omnivoice_prompt_text", voice["ref_text"])
                if engine == ENGINE_OMNIVOICE
                else voice["ref_text"],
                "seed": rng.randrange(1, 2**31 - 1),
            }
            if engine == ENGINE_OMNIVOICE:
                entry["language_id"] = self.omnivoice_language
            elif engine == ENGINE_QWEN3:
                entry["language_name"] = QWEN_LANGUAGE_NAMES[self.args.language]
            entries.append(entry)
            self.speed_by_path[(destination / f"{item_id}.wav").resolve()] = speed
        return entries

    def _repair_generated_corpus(
        self,
        engine: str,
        entries: list[dict],
        destination: Path,
        generation_command: list[str],
        prefix: str,
        *,
        speech_only: bool,
        input_flag: str,
        batch_flag: str | None = None,
    ) -> list[Path]:
        """Repair engine failures before they can enter the final corpus."""

        pending = {entry["id"]: entry for entry in entries}
        accepted_ids: set[str] = set()
        label = prefix or "main"
        retry_rounds = (
            OMNIVOICE_CORPUS_RETRY_ROUNDS if speech_only else MOSS_CORPUS_RETRY_ROUNDS
        )
        gate_name = "speech" if speech_only else "semantic"
        for qa_round in range(1, retry_rounds + 2):
            if engine == ENGINE_MOSS:
                for item_id in pending:
                    path = destination / f"{item_id}.wav"
                    if not path.is_file():
                        continue
                    qa_path = path.with_suffix(".qa.wav")
                    try:
                        subprocess.run(
                            [
                                self.args.ffmpeg,
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-y",
                                "-i",
                                str(path),
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                "-c:a",
                                "pcm_s16le",
                                str(qa_path),
                            ],
                            check=True,
                        )
                        qa_path.replace(path)
                    except subprocess.CalledProcessError:
                        qa_path.unlink(missing_ok=True)
            candidates = [
                {"id": item_id, "path": str(destination / f"{item_id}.wav")}
                for item_id in pending
                if (destination / f"{item_id}.wav").is_file()
            ]
            qa_input = self.build_dir / f"{engine}_{label}.{gate_name}-{qa_round}.jsonl"
            qa_output = self.build_dir / f"{engine}_{label}.{gate_name}-{qa_round}.results.jsonl"
            write_jsonl(qa_input, candidates)
            if candidates:
                qa_command = [
                    str(self._reference_qa_python()),
                    str(ROOT_DIR / "cli" / "tts_reference_qa.py"),
                    "--input-jsonl",
                    str(qa_input),
                    "--output-jsonl",
                    str(qa_output),
                    "--phrase",
                    self.spoken_phrase,
                    "--language",
                    self.args.language,
                    "--download-root",
                    str(self.data_dir / "auto_train_models"),
                ]
                if speech_only:
                    qa_command.append("--speech-only")
                run(
                    qa_command,
                    env=self.env,
                )
                round_accepted = {
                    result["id"]
                    for line in qa_output.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                    for result in (json.loads(line),)
                    if result.get("accepted")
                }
            else:
                round_accepted = set()
            accepted_ids.update(round_accepted)
            pending = {
                item_id: entry
                for item_id, entry in pending.items()
                if item_id not in round_accepted
            }
            if not pending:
                break
            for item_id in pending:
                (destination / f"{item_id}.wav").unlink(missing_ok=True)
            if qa_round > retry_rounds:
                log(
                    f"⚠️ {engine} {gate_name} gate dropped {len(pending)} persistent "
                    "bad decode(s); another engine will fill them."
                )
                break
            log(
                f"→ {engine} {gate_name} gate repair {qa_round}: regenerating "
                f"{len(pending)} bad decode(s)"
                + (" without STT" if speech_only else "")
            )
            retry_input = self.build_dir / f"{engine}_{label}.{gate_name}-retry-{qa_round}.jsonl"
            write_jsonl(retry_input, list(pending.values()))
            retry_command = list(generation_command)
            retry_command[retry_command.index(input_flag) + 1] = str(retry_input)
            if batch_flag:
                run_with_batch_retry(retry_command, batch_flag, env=self.env)
            else:
                run(retry_command, env=self.env)
        return [
            destination / f"{entry['id']}.wav"
            for entry in entries
            if entry["id"] in accepted_ids
        ]

    def generate_engine(self, engine: str, count: int, voices: list[dict], prefix: str = "") -> list[Path]:
        if count <= 0:
            return []
        destination_name = engine if not prefix else f"{engine}_{prefix.rstrip('_')}"
        destination = self.raw_dir / destination_name
        destination.mkdir(parents=True, exist_ok=True)
        requested = count + max(2, math.ceil(count * 0.01))

        if engine == ENGINE_PIPER:
            python = self.data_dir / ".venv" / "bin" / "python"
            generator = self.tools_dir / "piper-sample-generator" / "generate_samples.py"
            command = [
                str(python),
                str(generator),
                self.spoken_phrase,
                "--max-samples",
                str(requested),
                "--batch-size",
                str(self.args.batch_size),
                "--output-dir",
                str(destination),
                "--max-speakers",
                "100",
            ]
            for model in self.piper_models():
                command.extend(("--model", str(model)))
            run(command, env=self.env)
            for index, path in enumerate(sorted(destination.glob("*.wav"))):
                self.speed_by_path[path.resolve()] = SPEEDS[index % len(SPEEDS)]
            return sorted(destination.glob("*.wav"))

        entries = self.make_entries(engine, requested, voices, destination, prefix=prefix)
        input_path = self.build_dir / f"{engine}_{prefix or 'main'}.jsonl"
        write_jsonl(input_path, entries)
        python = self.ensure_environment(engine)
        if engine == ENGINE_OMNIVOICE:
            generation_command = [
                str(self.tts_envs / engine / "bin" / "omnivoice-infer-batch"),
                "--model",
                OMNIVOICE_MODEL,
                "--test_list",
                str(input_path),
                "--res_dir",
                str(destination),
                "--batch_size",
                str(max(1, min(self.args.batch_size, 8))),
                "--lang_id",
                self.omnivoice_language,
            ] + omnivoice_stability_args()
            run_with_batch_retry(
                generation_command,
                "--batch_size",
                env=self.env,
            )
            return self._repair_generated_corpus(
                engine,
                entries,
                destination,
                generation_command,
                prefix,
                speech_only=True,
                input_flag="--test_list",
                batch_flag="--batch_size",
            )
        elif engine == ENGINE_QWEN3:
            run_with_batch_retry(
                [
                    str(python),
                    str(ROOT_DIR / "cli" / "tts_qwen_worker.py"),
                    "--mode",
                    "generate",
                    "--input-jsonl",
                    str(input_path),
                    "--output-dir",
                    str(destination),
                    "--batch-size",
                    str(max(1, min(self.args.batch_size, 4))),
                ],
                "--batch-size",
                env=self.env,
            )
        elif engine == ENGINE_MOSS:
            generation_command = [
                str(python),
                str(ROOT_DIR / "cli" / "tts_moss_worker.py"),
                "--input-jsonl",
                str(input_path),
                "--output-dir",
                str(destination),
            ]
            run(generation_command, env=self.env)
            return self._repair_generated_corpus(
                engine,
                entries,
                destination,
                generation_command,
                prefix,
                speech_only=False,
                input_flag="--input-jsonl",
            )
        return sorted(destination.glob(f"{engine}_{prefix}*.wav"))

    def make_direct_entries(
        self,
        engine: str,
        count: int,
        destination: Path,
        reference_paths: list[Path],
        prefix: str = "",
    ) -> list[dict]:
        """Describe unique final candidates; no reusable 128-voice bank."""

        start = self.direct_attempt[engine]
        self.direct_attempt[engine] += count
        rng = random.Random(24051984 + start + sum(ord(ch) for ch in engine + prefix))
        descriptions = (
            qwen_descriptions(QWEN_LANGUAGE_NAMES[self.args.language], start + count)[start:]
            if engine == ENGINE_QWEN3
            else []
        )
        duration_scales = (0.82, 0.91, 1.0, 1.09, 1.18)
        entries = []
        for index in range(count):
            absolute_index = start + index
            item_id = f"{engine}_{prefix}{absolute_index:07d}"
            entry = {
                "id": item_id,
                "text": self.reference_text,
                "seed": rng.randrange(1, 2**31 - 1),
                "minimum_duration": self.minimum_duration,
                "maximum_duration": self.maximum_duration,
            }
            if engine == ENGINE_OMNIVOICE:
                entry.update(
                    {
                        "language_id": self.omnivoice_language,
                        "duration": round(
                            min(
                                self.maximum_duration * 0.9,
                                max(
                                    self.minimum_duration * 1.5,
                                    self.target_duration * duration_scales[absolute_index % len(duration_scales)],
                                ),
                            ),
                            3,
                        ),
                    }
                )
            elif engine == ENGINE_QWEN3:
                entry.update(
                    {
                        "language_name": QWEN_LANGUAGE_NAMES[self.args.language],
                        "instruct": descriptions[index],
                    }
                )
            elif engine == ENGINE_MOSS:
                if index >= len(reference_paths):
                    raise RuntimeError("MOSS direct corpus generation exhausted unique accepted references.")
                entry.update(
                    {
                        "ref_audio": str(reference_paths[index]),
                        "ref_text": self.reference_text,
                    }
                )
            entries.append(entry)
            self.speed_by_path[(destination / f"{item_id}.wav").resolve()] = 1.0
        return entries

    def generate_direct_engine(
        self,
        engine: str,
        count: int,
        reference_paths: list[Path],
        prefix: str = "",
    ) -> tuple[list[dict], list[Path]]:
        if count <= 0:
            return [], []
        destination_name = engine if not prefix else f"{engine}_{prefix.rstrip('_')}"
        destination = self.raw_dir / destination_name
        destination.mkdir(parents=True, exist_ok=True)
        requested = max(count, math.ceil(count * DIRECT_CANDIDATE_FACTORS[engine]))
        if engine == ENGINE_MOSS:
            requested = min(requested, len(reference_paths))
            if requested < count:
                log(
                    f"⚠️ MOSS has only {requested} unique accepted carrier(s) for "
                    f"{count} requested take(s); a direct provider will fill the remainder."
                )

        if engine == ENGINE_PIPER:
            command = [
                str(self.data_dir / ".venv" / "bin" / "python"),
                str(self.tools_dir / "piper-sample-generator" / "generate_samples.py"),
                self.spoken_phrase,
                "--max-samples",
                str(requested),
                "--batch-size",
                str(self.args.batch_size),
                "--output-dir",
                str(destination),
            ]
            for model in self.piper_models():
                command.extend(("--model", str(model)))
            run(command, env=self.env)
            paths = sorted(destination.glob("*.wav"))
            entries = [
                {
                    "id": path.stem,
                    "minimum_duration": self.minimum_duration,
                    "maximum_duration": self.maximum_duration,
                }
                for path in paths
            ]
            return entries, paths

        entries = self.make_direct_entries(engine, requested, destination, reference_paths, prefix)
        input_path = self.build_dir / f"{engine}_{prefix or 'main'}.direct.jsonl"
        write_jsonl(input_path, entries)
        python = self.ensure_environment(engine)
        if engine == ENGINE_OMNIVOICE:
            command = [
                str(self.tts_envs / engine / "bin" / "omnivoice-infer-batch"),
                "--model",
                OMNIVOICE_MODEL,
                "--test_list",
                str(input_path),
                "--res_dir",
                str(destination),
                "--batch_size",
                str(max(1, min(self.args.batch_size, 8))),
                "--lang_id",
                self.omnivoice_language,
                "--num_step",
                "32",
                "--denoise",
                "True",
                "--postprocess_output",
                "True",
            ] + omnivoice_stability_args()
            run_with_batch_retry(command, "--batch_size", env=self.env)
        elif engine == ENGINE_QWEN3:
            command = [
                str(python),
                str(ROOT_DIR / "cli" / "tts_qwen_worker.py"),
                "--mode",
                "direct",
                "--input-jsonl",
                str(input_path),
                "--output-dir",
                str(destination),
                "--batch-size",
                str(max(1, min(self.args.batch_size, 4))),
            ]
            run_with_batch_retry(command, "--batch-size", env=self.env)
        elif engine == ENGINE_MOSS:
            run(
                [
                    str(python),
                    str(ROOT_DIR / "cli" / "tts_moss_worker.py"),
                    "--input-jsonl",
                    str(input_path),
                    "--output-dir",
                    str(destination),
                ],
                env=self.env,
            )
        return entries, [
            destination / f"{entry['id']}.wav"
            for entry in entries
            if (destination / f"{entry['id']}.wav").is_file()
        ]

    def qualify_direct_candidates(
        self,
        engine: str,
        entries: list[dict],
        paths: list[Path],
        prefix: str = "",
    ) -> list[Path]:
        entries_by_id = {entry["id"]: entry for entry in entries}
        candidates = []
        for path in paths:
            entry = entries_by_id.get(path.stem, {})
            candidates.append(
                {
                    "id": path.stem,
                    "path": str(path),
                    "minimum_duration": entry.get("minimum_duration", self.minimum_duration),
                    "maximum_duration": entry.get("maximum_duration", self.maximum_duration),
                }
            )
        if not candidates:
            return []
        label = prefix or "main"
        qa_input = self.build_dir / f"{engine}_{label}.direct-qa.jsonl"
        qa_output = self.build_dir / f"{engine}_{label}.direct-qa.results.jsonl"
        write_jsonl(qa_input, candidates)
        run(
            [
                str(self._reference_qa_python()),
                str(ROOT_DIR / "cli" / "tts_reference_qa.py"),
                "--input-jsonl",
                str(qa_input),
                "--output-jsonl",
                str(qa_output),
                "--phrase",
                self.spoken_phrase,
                "--language",
                self.args.language,
                "--download-root",
                str(self.data_dir / "auto_train_models"),
                "--speech-only",
                "--profile",
                engine,
            ],
            env=self.env,
        )
        results = [
            json.loads(line)
            for line in qa_output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        accepted_ids = {result["id"] for result in results if result.get("accepted")}
        rejected = Counter(result.get("reason", "unknown") for result in results if not result.get("accepted"))
        if rejected:
            log(
                f"⚠️ {engine} safety gate rejected {sum(rejected.values())} candidate(s): "
                + ", ".join(f"{reason}={count}" for reason, count in sorted(rejected.items()))
            )
        return [path for path in paths if path.stem in accepted_ids]

    def normalize(self, paths: list[Path], start_index: int, limit: int) -> list[Path]:
        accepted = []
        self.final_dir.mkdir(parents=True, exist_ok=True)
        candidate_count = len(paths)
        log(f"→ Normalizing up to {limit} accepted clip(s) from {candidate_count} candidate(s)")
        for processed, path in enumerate(paths, start=1):
            if len(accepted) >= limit:
                break
            final_path = self.final_dir / f"{start_index + len(accepted)}.wav"
            speed = self.speed_by_path.get(path.resolve(), 1.0)
            temp_path = final_path.with_suffix(".tmp.wav")
            command = [
                self.args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-af",
                f"atempo={speed}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temp_path),
            ]
            return_code = run_normalization_ffmpeg(
                command,
                timeout=NORMALIZATION_TIMEOUT_SECONDS,
            )
            converted = return_code == 0
            if return_code is None:
                temp_path.unlink(missing_ok=True)
                log(
                    f"⚠️ Normalization timed out after "
                    f"{NORMALIZATION_TIMEOUT_SECONDS:g}s; skipping {path.name}"
                )
            elif return_code != 0:
                temp_path.unlink(missing_ok=True)
                log(f"⚠️ ffmpeg rejected {path.name} (exit {return_code}); skipping it")

            if converted:
                digest = hashlib.sha256(temp_path.read_bytes()).hexdigest() if temp_path.is_file() else ""
                if valid_sample(temp_path) and digest and digest not in self.accepted_hashes:
                    temp_path.replace(final_path)
                    self.accepted_hashes.add(digest)
                    accepted.append(final_path)
                else:
                    temp_path.unlink(missing_ok=True)

            if (
                processed % NORMALIZATION_PROGRESS_INTERVAL == 0
                or processed == candidate_count
                or len(accepted) >= limit
            ):
                log(
                    f"Normalization progress: {len(accepted)}/{limit} accepted "
                    f"({processed}/{candidate_count} candidate(s) checked)"
                )
        return accepted

    def generate(self) -> None:
        if self.cache_hit():
            log("✅ Reusing the matching direct-generated TTS corpus.")
            return

        engines = self.engines()
        if not engines:
            raise RuntimeError(
                f"No TTS engine is available for language={self.args.language} mode={self.args.tts_mode}."
            )
        shutil.rmtree(self.build_dir, ignore_errors=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        plan = distribute_samples(self.args.samples, engines)
        log(f"===== Direct TTS corpus plan ({self.args.tts_mode}, {self.args.language}) =====")
        for engine, count in plan.items():
            log(f"   {engine}: {count} sample(s)")
        log(
            f"   safety duration: {self.minimum_duration:.2f}–{self.maximum_duration:.2f}s; "
            "static, silence, clipping, rambling, and exact duplicates are rejected"
        )

        accepted: list[Path] = []
        successful_engines: list[str] = []
        ordered_engines = [
            engine
            for engine in (ENGINE_QWEN3, ENGINE_PIPER, ENGINE_OMNIVOICE, ENGINE_MOSS)
            if engine in plan
        ]
        for engine in ordered_engines:
            count = plan[engine]
            try:
                entries, raw_paths = self.generate_direct_engine(engine, count, list(accepted))
                qualified_paths = self.qualify_direct_candidates(engine, entries, raw_paths)
                requested_accepts = min(count, self.args.samples - len(accepted))
                normalized = self.normalize(qualified_paths, len(accepted), requested_accepts)
                accepted.extend(normalized)
                self.actual_counts[engine] = len(normalized)
                if normalized:
                    successful_engines.append(engine)
                log(f"✅ {engine}: accepted {len(normalized)} normalized sample(s)")
            except Exception as error:
                self.actual_counts[engine] = 0
                log(f"⚠️ {engine} generation failed; another engine will fill its share: {error}")

        missing = self.args.samples - len(accepted)
        fallback_candidates = [
            engine
            for engine in (ENGINE_QWEN3, ENGINE_PIPER, ENGINE_OMNIVOICE, ENGINE_MOSS)
            if engine in successful_engines
            or (engine == ENGINE_PIPER and engine in engines and self.piper_available())
        ]
        for attempt in range(6):
            if missing <= 0 or not fallback_candidates:
                break
            engine = fallback_candidates[attempt % len(fallback_candidates)]
            log(f"→ Filling {missing} rejected/missing sample(s) with {engine}")
            try:
                entries, raw_paths = self.generate_direct_engine(
                    engine,
                    missing,
                    list(accepted),
                    prefix=f"fallback{attempt}_",
                )
                qualified_paths = self.qualify_direct_candidates(
                    engine,
                    entries,
                    raw_paths,
                    prefix=f"fallback{attempt}_",
                )
                normalized = self.normalize(qualified_paths, len(accepted), missing)
                accepted.extend(normalized)
                self.actual_counts[engine] = self.actual_counts.get(engine, 0) + len(normalized)
            except Exception as error:
                log(f"⚠️ {engine} fallback failed: {error}")
            missing = self.args.samples - len(accepted)

        if len(accepted) < self.args.samples:
            raise RuntimeError(
                f"Only {len(accepted)} of {self.args.samples} samples passed normalization and QA."
            )

        # Remove any accepted overage and ensure contiguous numeric file names.
        for path in list(self.final_dir.glob("*.wav")):
            try:
                index = int(path.stem)
            except ValueError:
                path.unlink(missing_ok=True)
                continue
            if index >= self.args.samples:
                path.unlink(missing_ok=True)

        manifest = {
            "signature": self.signature(),
            "planned_counts": plan,
            "actual_counts": self.actual_counts,
            "voice_bank": "",
            "generation_strategy": {
                "direct_final_candidates": True,
                "reusable_profile_bank": False,
                "moss_unique_accepted_carriers": True,
                "piper_all_model_speakers": True,
            },
            "qa": {
                "audio_format": "16 kHz mono PCM16 WAV",
                "duration_seconds": [self.minimum_duration, self.maximum_duration],
                "minimum_rms": 0.004,
                "maximum_clipped_ratio": 0.01,
                "provider_specific_acoustic_gate": True,
                "static_and_broadband_noise_gate": True,
                "exact_duplicate_gate": True,
                "qwen_max_acoustic_tokens": 48,
                "moss_max_acoustic_frames": 64,
                "omnivoice_fixed_short_duration": True,
            },
        }
        (self.final_dir / ".generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.move(str(self.final_dir), str(self.output_dir))
        shutil.rmtree(self.build_dir, ignore_errors=True)
        log(f"✅ Generated {self.args.samples} ensemble sample(s) in {self.output_dir}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("phrase")
    result.add_argument("--language", default="en")
    result.add_argument("--tts-mode", default=DEFAULT_TTS_MODE)
    result.add_argument("--samples", type=int, default=50000)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument(
        "--voice-count",
        type=int,
        default=128,
        help=argparse.SUPPRESS,  # accepted for compatibility; direct generation ignores it
    )
    result.add_argument("--data-dir", type=Path, default=Path("/data"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    args.language = args.language.strip().lower().replace("-", "_")
    args.tts_mode = normalize_tts_mode(args.tts_mode)
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.voice_count < 1:
        raise SystemExit("--voice-count must be positive")
    generator = Generator(args)
    if args.dry_run:
        engines = generator.engines()
        print(
            json.dumps(
                {
                    "signature": generator.signature(),
                    "plan": distribute_samples(args.samples, engines),
                    "piper_available": generator.piper_available(),
                },
                indent=2,
            )
        )
        return 0

    lock_dir = generator.data_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "tts-gpu.lock").open("w", encoding="utf-8") as lock_file:
        log("→ Waiting for the TTS GPU lock")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        generator.hf_home.mkdir(parents=True, exist_ok=True)
        generator.generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
