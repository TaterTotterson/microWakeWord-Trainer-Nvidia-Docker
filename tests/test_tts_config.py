from __future__ import annotations

import unittest

from tts_config import (
    ENGINE_MOSS,
    ENGINE_OMNIVOICE,
    ENGINE_PIPER,
    ENGINE_QWEN3,
    distribute_samples,
    engines_for_language,
    language_for_engine,
    normalize_tts_mode,
    quality_for_engines,
)


class TtsConfigTests(unittest.TestCase):
    def test_recommended_languages_use_all_modern_engines(self) -> None:
        self.assertEqual(
            engines_for_language("en", "modern"),
            [ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS],
        )
        self.assertEqual(
            quality_for_engines(engines_for_language("fr", "modern")),
            "recommended",
        )

    def test_broad_language_coverage_routes_through_omnivoice(self) -> None:
        self.assertEqual(engines_for_language("zu", "modern"), [ENGINE_OMNIVOICE])
        self.assertEqual(quality_for_engines([ENGINE_OMNIVOICE]), "experimental")

    def test_hybrid_and_legacy_modes_require_available_piper(self) -> None:
        self.assertEqual(
            engines_for_language("en", "hybrid", piper_available=True),
            [ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS, ENGINE_PIPER],
        )
        self.assertEqual(engines_for_language("en", "piper"), [])
        self.assertEqual(
            engines_for_language("en", "piper", piper_available=True),
            [ENGINE_PIPER],
        )

    def test_sample_distribution_is_exact_and_deterministic(self) -> None:
        self.assertEqual(
            distribute_samples(10, [ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS]),
            {ENGINE_OMNIVOICE: 4, ENGINE_QWEN3: 3, ENGINE_MOSS: 3},
        )
        self.assertEqual(sum(distribute_samples(50000, ["a", "b", "c"]).values()), 50000)

    def test_invalid_mode_falls_back_to_four_provider_route(self) -> None:
        self.assertEqual(normalize_tts_mode("unknown"), "hybrid")

    def test_common_language_aliases_use_model_catalog_ids(self) -> None:
        self.assertEqual(language_for_engine(ENGINE_OMNIVOICE, "ar"), "arb")
        self.assertEqual(language_for_engine(ENGINE_OMNIVOICE, "ne"), "npi")
        self.assertEqual(language_for_engine(ENGINE_MOSS, "ar"), "ar")


if __name__ == "__main__":
    unittest.main()
