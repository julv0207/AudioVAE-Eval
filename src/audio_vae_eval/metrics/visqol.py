from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio

MOS_PATTERNS = (
    re.compile(r"MOS-LQO[^0-9-]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"moslqo[^0-9-]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
)


def _write_visqol_input(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 48_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 48_000)
    torchaudio.save(str(path), audio.cpu(), 48_000, encoding="PCM_S", bits_per_sample=16)


def visqol_moslqo(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    executable: str | Path,
) -> float:
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"ViSQOL executable not found at {executable}; run bootstrap --build-visqol"
        )
    with tempfile.TemporaryDirectory(prefix="audio-vae-visqol-") as directory:
        directory = Path(directory)
        reference_path = directory / "reference.wav"
        estimate_path = directory / "estimate.wav"
        _write_visqol_input(reference_path, reference, sample_rate)
        _write_visqol_input(estimate_path, estimate, sample_rate)
        command = [
            str(executable),
            "--reference_file",
            str(reference_path),
            "--degraded_file",
            str(estimate_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode:
            raise RuntimeError(f"ViSQOL failed ({completed.returncode}): {output.strip()}")
        for pattern in MOS_PATTERNS:
            match = pattern.search(output)
            if match:
                return float(match.group(1))
    raise RuntimeError(f"Could not parse ViSQOL MOS-LQO output: {output.strip()}")
