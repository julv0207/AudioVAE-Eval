from __future__ import annotations

from collections.abc import Iterable

import torch
import torchaudio

EPS = 1e-10


def _batched(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio[None, None]
    elif audio.ndim == 2:
        audio = audio[None]
    if audio.ndim != 3:
        raise ValueError(f"Expected (time), (channels,time), or (batch,channels,time): {audio.shape}")
    return audio


def magnitude_stft(
    audio: torch.Tensor,
    n_fft: int,
    *,
    hop_length: int | None = None,
    center: bool = True,
) -> torch.Tensor:
    audio = _batched(audio)
    batch, channels, samples = audio.shape
    flattened = audio.reshape(batch * channels, samples)
    spectrum = torch.stft(
        flattened,
        n_fft=n_fft,
        hop_length=hop_length or n_fft // 4,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype),
        center=center,
        return_complex=True,
    )
    return spectrum.abs().reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])


def stft_log1p(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    fft_sizes: Iterable[int] = (128, 256, 512, 1024, 2048, 4096),
) -> torch.Tensor:
    losses = []
    for n_fft in fft_sizes:
        ref = torch.log1p(magnitude_stft(reference, n_fft))
        est = torch.log1p(magnitude_stft(estimate, n_fft))
        losses.append(torch.mean(torch.abs(ref - est)))
    return torch.stack(losses).mean()


def mel_log1p(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    fft_sizes: Iterable[int] = (1024, 2048, 4096),
    n_mels: int = 64,
) -> torch.Tensor:
    losses = []
    for n_fft in fft_sizes:
        ref = magnitude_stft(reference, n_fft)
        est = magnitude_stft(estimate, n_fft)
        filters = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        ).to(device=ref.device, dtype=ref.dtype)
        scale = n_mels / (n_fft // 2 + 1)
        ref_mel = torch.einsum("bcft,fm->bcmt", ref, filters) * scale
        est_mel = torch.einsum("bcft,fm->bcmt", est, filters) * scale
        losses.append(torch.mean(torch.abs(torch.log1p(ref_mel) - torch.log1p(est_mel))))
    return torch.stack(losses).mean()


def _band_mask(
    sample_rate: int,
    n_fft: int,
    min_hz: float | None,
    max_hz: float | None,
    min_inclusive: bool = True,
    max_inclusive: bool = False,
) -> torch.Tensor:
    frequencies = torch.fft.rfftfreq(n_fft, d=1 / sample_rate)
    mask = torch.ones_like(frequencies, dtype=torch.bool)
    if min_hz is not None:
        mask &= frequencies >= min_hz if min_inclusive else frequencies > min_hz
    if max_hz is not None:
        mask &= frequencies <= max_hz if max_inclusive else frequencies < max_hz
    return mask


def band_lsd(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    min_hz: float | None,
    max_hz: float | None,
    n_fft: int = 2048,
) -> torch.Tensor | None:
    mask = _band_mask(sample_rate, n_fft, min_hz, max_hz)
    if not bool(mask.any()):
        return None
    ref_full = magnitude_stft(reference, n_fft, hop_length=n_fft // 4, center=False)
    est_full = magnitude_stft(estimate, n_fft, hop_length=n_fft // 4, center=False)
    mask = mask.to(ref_full.device)
    ref = ref_full[..., mask, :].square().clamp_min(EPS)
    est = est_full[..., mask, :].square().clamp_min(EPS)
    squared_db_error = (10 * torch.log10(ref) - 10 * torch.log10(est)).square()
    return squared_db_error.mean(dim=-2).sqrt().mean()


def band_energy_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    min_hz: float | None,
    max_hz: float | None,
    n_fft: int = 2048,
) -> torch.Tensor | None:
    mask = _band_mask(sample_rate, n_fft, min_hz, max_hz)
    if not bool(mask.any()):
        return None
    ref_full = magnitude_stft(reference, n_fft)
    est_full = magnitude_stft(estimate, n_fft)
    mask = mask.to(ref_full.device)
    ref_energy = ref_full[..., mask, :].square().sum(dim=-2)
    est_energy = est_full[..., mask, :].square().sum(dim=-2)
    error_db = 10 * torch.log10((est_energy + EPS) / (ref_energy + EPS))
    return error_db.abs().mean()


def multiresolution_stft_distance(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    min_hz: float | None = None,
    max_hz: float | None = None,
    fft_sizes: Iterable[int] = (64, 128, 256, 512, 1024, 2048),
    min_inclusive: bool = True,
    max_inclusive: bool = False,
) -> torch.Tensor | None:
    losses = []
    for n_fft in fft_sizes:
        mask = _band_mask(
            sample_rate,
            n_fft,
            min_hz,
            max_hz,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        )
        if not bool(mask.any()):
            continue
        ref_full = magnitude_stft(reference, n_fft)
        est_full = magnitude_stft(estimate, n_fft)
        mask = mask.to(ref_full.device)
        ref = ref_full[..., mask, :]
        est = est_full[..., mask, :]
        losses.append(torch.mean(torch.abs(ref - est)))
    return torch.stack(losses).mean() if losses else None
