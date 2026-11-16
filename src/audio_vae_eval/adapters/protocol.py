from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from audio_vae_eval.registry import get_model_spec

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ReconstructionResult:
    model: str
    reference_path: Path
    reconstruction_path: Path
    metadata_path: Path
    metadata: dict[str, object]


def environment_python(model: str) -> Path:
    spec = get_model_spec(model)
    python = ROOT / ".venvs" / spec.environment / "bin" / "python"
    if python.exists():
        return python
    if os.environ.get("AUDIO_VAE_EVAL_ALLOW_CURRENT_PYTHON") == "1":
        return Path(os.sys.executable)
    raise FileNotFoundError(
        f"Missing environment {python}. Run `audio-vae-eval bootstrap` first."
    )


def reconstruct(
    model: str,
    input_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    dtype: str = "float32",
) -> ReconstructionResult:
    get_model_spec(model)
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    reconstruction = output_dir / f"{stem}.wav"
    metadata_path = output_dir / f"{stem}.json"
    request = {
        "model": model,
        "input_path": str(input_path),
        "output_path": str(reconstruction),
        "metadata_path": str(metadata_path),
        "device": device,
        "dtype": dtype,
        "root": str(ROOT),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(request, stream)
        request_path = Path(stream.name)
    try:
        command = [
            str(environment_python(model)),
            "-m",
            "audio_vae_eval.backends.runner",
            "--request",
            str(request_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"{model} reconstruction failed ({completed.returncode}):\n"
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
    finally:
        request_path.unlink(missing_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ReconstructionResult(
        model=model,
        reference_path=input_path,
        reconstruction_path=reconstruction,
        metadata_path=metadata_path,
        metadata=metadata,
    )
