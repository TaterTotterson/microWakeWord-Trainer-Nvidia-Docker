"""Shared modern-TTS catalog and routing helpers.

This module intentionally has no third-party dependencies.  It is imported by
the web server, the shell-facing generator, and unit tests before any of the
large model environments have been installed.
"""

from __future__ import annotations

from typing import Iterable


TTS_MODE_MODERN = "modern"
TTS_MODE_HYBRID = "hybrid"
TTS_MODE_PIPER = "piper"
TTS_MODES = (TTS_MODE_MODERN, TTS_MODE_HYBRID, TTS_MODE_PIPER)
DEFAULT_TTS_MODE = TTS_MODE_HYBRID

ENGINE_OMNIVOICE = "omnivoice"
ENGINE_QWEN3 = "qwen3"
ENGINE_MOSS = "moss"
ENGINE_PIPER = "piper"
MODERN_ENGINES = (ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS)

# Friendly/common codes whose OmniVoice IDs follow the model's catalog IDs.
OMNIVOICE_LANGUAGE_ALIASES = {
    "ar": "arb",  # Standard Arabic
    "ne": "npi",  # Nepali
}


QWEN_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# The upstream MOSS-TTS-Nano README calls this a 20-language list, although
# the published table currently contains the 19 concrete entries below.
MOSS_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "it": "Italian",
    "hu": "Hungarian",
    "ko": "Korean",
    "ru": "Russian",
    "fa": "Persian (Farsi)",
    "ar": "Arabic",
    "pl": "Polish",
    "pt": "Portuguese",
    "cs": "Czech",
    "da": "Danish",
    "sv": "Swedish",
    "el": "Greek",
    "tr": "Turkish",
}

# Used when the live OmniVoice catalog has not been downloaded yet.  The web
# server expands this to the full upstream catalog (currently 646 languages)
# and persists it under /data/.cache.
COMMON_OMNIVOICE_LANGUAGES = {
    **QWEN_LANGUAGES,
    **MOSS_LANGUAGES,
    "af": "Afrikaans",
    "am": "Amharic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cy": "Welsh",
    "et": "Estonian",
    "eu": "Basque",
    "fi": "Finnish",
    "fil": "Filipino",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ka": "Georgian",
    "kk": "Kazakh",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "yo": "Yoruba",
    "zu": "Zulu",
}

QWEN_LANGUAGE_NAMES = {code: name for code, name in QWEN_LANGUAGES.items()}


def parse_omnivoice_catalog(markdown: str) -> dict[str, dict[str, object]]:
    """Parse the upstream Markdown language table without third-party packages."""

    entries: dict[str, dict[str, object]] = {}
    for line in str(markdown or "").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        name, code, iso_code, duration_text = cells[1:5]
        code = code.strip().lower().replace("-", "_")
        if not code or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code):
            continue
        try:
            duration_hours = float(duration_text.replace(",", ""))
        except ValueError:
            duration_hours = 0.0
        entries[code] = {
            "name": name or code.upper(),
            "iso_639_3": iso_code,
            "duration_hours": duration_hours,
        }
    return entries


def normalize_tts_mode(value: object) -> str:
    token = str(value or DEFAULT_TTS_MODE).strip().lower().replace("-", "_")
    return token if token in TTS_MODES else DEFAULT_TTS_MODE


def language_for_engine(engine: str, language: str) -> str:
    code = str(language or "en").strip().lower().replace("-", "_")
    if engine == ENGINE_OMNIVOICE:
        return OMNIVOICE_LANGUAGE_ALIASES.get(code, code)
    return code


def modern_engines_for_language(language: str) -> list[str]:
    """Return modern engines ordered from broadest to most specialized."""

    code = str(language or "en").strip().lower().replace("-", "_")
    engines = [ENGINE_OMNIVOICE]
    if code in QWEN_LANGUAGES:
        engines.append(ENGINE_QWEN3)
    if code in MOSS_LANGUAGES:
        engines.append(ENGINE_MOSS)
    return engines


def engines_for_language(
    language: str,
    mode: object = DEFAULT_TTS_MODE,
    *,
    piper_available: bool = False,
) -> list[str]:
    selected_mode = normalize_tts_mode(mode)
    if selected_mode == TTS_MODE_PIPER:
        return [ENGINE_PIPER] if piper_available else []

    engines = modern_engines_for_language(language)
    if selected_mode == TTS_MODE_HYBRID and piper_available:
        engines.append(ENGINE_PIPER)
    return engines


def quality_for_engines(engines: Iterable[str]) -> str:
    engine_set = set(engines)
    if ENGINE_QWEN3 in engine_set and ENGINE_MOSS in engine_set:
        return "recommended"
    if ENGINE_MOSS in engine_set:
        return "supported"
    if ENGINE_OMNIVOICE in engine_set:
        return "experimental"
    return "legacy"


def distribute_samples(total: int, engines: Iterable[str]) -> dict[str, int]:
    """Distribute an exact sample total as evenly as possible."""

    ordered = list(dict.fromkeys(str(engine) for engine in engines if engine))
    if total < 0:
        raise ValueError("total must be non-negative")
    if not ordered:
        if total:
            raise ValueError("at least one engine is required")
        return {}

    quotient, remainder = divmod(total, len(ordered))
    return {
        engine: quotient + (1 if index < remainder else 0)
        for index, engine in enumerate(ordered)
    }
