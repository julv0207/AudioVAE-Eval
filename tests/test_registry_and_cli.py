from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from audio_vae_eval.cli import discover_audio
from audio_vae_eval.registry import DEFAULT_MODELS, MODEL_REGISTRY, get_model_spec


class RegistryAndCliTests(unittest.TestCase):
    def test_six_default_models_are_registered(self) -> None:
        self.assertEqual(len(DEFAULT_MODELS), 6)
        self.assertLessEqual(set(DEFAULT_MODELS), MODEL_REGISTRY.keys())
        self.assertTrue(
            math.isclose(
                get_model_spec("stable-audio-open").latent_frame_rate, 44_100 / 2048
            )
        )
        self.assertEqual(get_model_spec("same-s").temporal_stride, 4096)
        self.assertEqual(get_model_spec("woosh-ae").latent_frame_rate, 100)

    def test_model_shape_contract_metadata(self) -> None:
        expected = {
            "audioldm2": (16_000, 1, 8),
            "mmaudio-44k": (44_100, 1, 40),
            "stable-audio-open": (44_100, 2, 64),
            "same-s": (44_100, 2, 256),
            "same-l": (44_100, 2, 256),
            "woosh-ae": (48_000, 1, 128),
        }
        for name, expected_shape in expected.items():
            spec = get_model_spec(name)
            actual = (spec.sample_rate, spec.channels, spec.latent_channels)
            self.assertEqual(actual, expected_shape)

    def test_jsonl_manifest_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.touch()
            manifest = root / "files.jsonl"
            manifest.write_text(json.dumps({"path": "sample.wav"}) + "\n", encoding="utf-8")
            self.assertEqual(discover_audio(manifest, [".wav"]), [audio.resolve()])
