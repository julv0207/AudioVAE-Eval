from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from audio_vae_eval.audio import align_pair
from audio_vae_eval.metrics.spectral import (
    band_energy_error,
    band_lsd,
    mel_log1p,
    multiresolution_stft_distance,
    stft_log1p,
)
from audio_vae_eval.metrics.visqol import visqol_moslqo


def _number(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().cpu())


def evaluate_pair(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, float | None]:
    reference, estimate = align_pair(reference.float(), estimate.float())
    metrics = config["metrics"]
    result: dict[str, float | None] = {}
    zero_mean = bool(metrics.get("si_sdr", {}).get("zero_mean", False))
    result["si_sdr"] = float(
        scale_invariant_signal_distortion_ratio(
            estimate, reference, zero_mean=zero_mean
        ).mean()
    )
    result["stft_log1p"] = _number(
        stft_log1p(reference, estimate, metrics["stft_log1p"]["fft_sizes"])
    )
    mel_config = metrics["mel_log1p"]
    result["mel_log1p"] = _number(
        mel_log1p(
            reference,
            estimate,
            sample_rate,
            mel_config["fft_sizes"],
            int(mel_config["n_mels"]),
        )
    )
    fft_sizes = metrics["spectral"]["fft_sizes"]
    result["stft_lf"] = _number(
        multiresolution_stft_distance(
            reference, estimate, sample_rate, max_hz=2_000, fft_sizes=fft_sizes
        )
    )
    for cutoff_hz in metrics["spectral"].get("hf_cutoffs_hz", (8_000, 16_000)):
        cutoff_hz = float(cutoff_hz)
        label = (
            f"{int(cutoff_hz / 1000)}"
            if cutoff_hz % 1000 == 0
            else f"{cutoff_hz:g}hz"
        )
        result[f"stft_hf{label}"] = _number(
            multiresolution_stft_distance(
                reference,
                estimate,
                sample_rate,
                min_hz=cutoff_hz,
                fft_sizes=fft_sizes,
                min_inclusive=False,
            )
        )
    for band in metrics["bands"]:
        name = str(band["name"])
        min_hz = band.get("min_hz")
        max_hz = band.get("max_hz")
        result[f"lsd/{name}"] = _number(
            band_lsd(reference, estimate, sample_rate, min_hz, max_hz)
        )
        result[f"energy_error_db/{name}"] = _number(
            band_energy_error(reference, estimate, sample_rate, min_hz, max_hz)
        )

    visqol = metrics.get("visqol", {})
    if visqol.get("enabled", True):
        executable = Path(visqol["executable"])
        if not executable.is_absolute() and project_root is not None:
            executable = Path(project_root) / executable
        result["visqol_moslqo"] = visqol_moslqo(
            reference, estimate, sample_rate, executable
        )
    return result
