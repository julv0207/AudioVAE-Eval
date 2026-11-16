#!/usr/bin/env python3
"""Run a short encode/decode smoke test for installed model environments."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from audio_vae_eval.adapters import reconstruct
from audio_vae_eval.audio import save_audio
from audio_vae_eval.registry import DEFAULT_MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=0.25)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="audio-vae-smoke-") as directory:
        root = Path(directory)
        sample_rate = 48_000
        time = torch.arange(int(sample_rate * args.seconds)) / sample_rate
        waveform = (0.1 * torch.sin(2 * torch.pi * 440 * time)).unsqueeze(0)
        source = root / "sine.wav"
        save_audio(source, waveform, sample_rate)
        failed = False
        for model in args.models:
            try:
                result = reconstruct(model, source, root / model, args.device)
                print(
                    json.dumps(
                        {
                            "model": model,
                            "status": "ok",
                            "latent_shape": result.metadata["latent_shape"],
                            "output_samples": result.metadata["output_samples"],
                        }
                    )
                )
            except Exception as error:
                failed = True
                print(
                    json.dumps(
                        {
                            "model": model,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                )
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
