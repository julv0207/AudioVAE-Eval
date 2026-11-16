from __future__ import annotations

import unittest

import torch

from audio_vae_eval.metrics.spectral import (
    band_energy_error,
    band_lsd,
    mel_log1p,
    multiresolution_stft_distance,
    stft_log1p,
)


def sine(sample_rate: int, frequency: float, seconds: float = 0.3) -> torch.Tensor:
    time = torch.arange(int(sample_rate * seconds)) / sample_rate
    return torch.sin(2 * torch.pi * frequency * time).unsqueeze(0)


class SpectralMetricTests(unittest.TestCase):
    def test_identical_signals_have_zero_spectral_error(self) -> None:
        audio = sine(48_000, 440)
        self.assertTrue(torch.allclose(stft_log1p(audio, audio, (128, 512)), torch.tensor(0.0)))
        self.assertTrue(
            torch.allclose(mel_log1p(audio, audio, 48_000, (1024,), 64), torch.tensor(0.0))
        )
        self.assertTrue(
            torch.allclose(
                band_lsd(audio, audio, 48_000, 0, 2_000, n_fft=512), torch.tensor(0.0)
            )
        )
        self.assertTrue(
            torch.allclose(
                band_energy_error(audio, audio, 48_000, 0, 2_000, n_fft=512),
                torch.tensor(0.0),
            )
        )

    def test_hf16_is_unavailable_at_16khz(self) -> None:
        audio = sine(16_000, 440)
        result = multiresolution_stft_distance(
            audio, audio, 16_000, min_hz=16_000, min_inclusive=False, fft_sizes=(128, 512)
        )
        self.assertIsNone(result)

    def test_band_errors_detect_changed_signal(self) -> None:
        reference = sine(48_000, 1_000)
        estimate = reference * 0.5
        energy = band_energy_error(reference, estimate, 48_000, 0, 2_000, n_fft=1024)
        lsd = band_lsd(reference, estimate, 48_000, 0, 2_000, n_fft=1024)
        self.assertIsNotNone(energy)
        self.assertIsNotNone(lsd)
        self.assertTrue(5.5 < float(energy) < 6.5)
        self.assertGreater(float(lsd), 0)
