from __future__ import annotations

import math
from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: str | Path,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> tuple[torch.Tensor, int]:
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.float()
    if sample_rate is not None and source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
        source_rate = sample_rate
    if channels == 1 and waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif channels == 2:
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
    return waveform.contiguous(), source_rate


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(
        str(destination), waveform, sample_rate, encoding="PCM_F", bits_per_sample=32
    )


def pad_to_multiple(waveform: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    original_length = waveform.shape[-1]
    padded_length = math.ceil(original_length / multiple) * multiple
    return torch.nn.functional.pad(waveform, (0, padded_length - original_length)), original_length


def align_pair(reference: torch.Tensor, estimate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)
    channels = min(reference.shape[-2], estimate.shape[-2])
    samples = min(reference.shape[-1], estimate.shape[-1])
    return reference[:channels, :samples], estimate[:channels, :samples]
