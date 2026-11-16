#!/usr/bin/env python3
"""Fetch official sources/checkpoints and create isolated uv environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
CHECKPOINTS = ROOT / "checkpoints"
ENVS = ROOT / ".venvs"

REPOSITORIES = {
    "audioldm2": "https://github.com/haoheliu/AudioLDM2.git",
    "mmaudio": "https://github.com/hkchengrex/MMAudio.git",
    "stable-audio-3": "https://github.com/Stability-AI/stable-audio-3.git",
    "woosh": "https://github.com/SonyResearch/Woosh.git",
    "visqol": "https://github.com/google/visqol.git",
}

LICENSES = {
    "framework": "MIT",
    "audioldm2": "CC BY-NC-SA 4.0 (weights/root project)",
    "mmaudio": "MIT (code), CC BY-NC 4.0 (weights)",
    "stable-audio-open": "Stability AI Community License (weights)",
    "same-s": "Stability AI Community License",
    "same-l": "Stability AI Community License",
    "woosh": "MIT/Apache-2.0 (code), CC BY-NC (weights)",
    "visqol": "Apache-2.0",
}

HF_SNAPSHOTS = {
    "audioldm2": (
        "cvssp/audioldm2",
        ["vae/*", "vocoder/*", "model_index.json"],
    ),
    "mmaudio": (
        "hkchengrex/MMAudio",
        ["ext_weights/v1-16.pth", "ext_weights/v1-44.pth", "ext_weights/best_netG.pt"],
    ),
    "mmaudio-bigvgan": (
        "nvidia/bigvgan_v2_44khz_128band_512x",
        ["*.json", "*.pt", "*.safetensors"],
    ),
    "same-s": ("stabilityai/SAME-S", ["model.safetensors", "model_config.json", "*.md"]),
    "same-l": ("stabilityai/SAME-L", ["model.safetensors", "model_config.json", "*.md"]),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def clone_sources(names: list[str]) -> None:
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = THIRD_PARTY / name
        if destination.exists():
            print(f"source exists: {destination}")
            continue
        run(["git", "clone", "--depth", "1", REPOSITORIES[name], str(destination)])


def download_hf(names: list[str]) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for name in names:
        repo_id, patterns = HF_SNAPSHOTS[name]
        destination = CHECKPOINTS / name
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            allow_patterns=patterns,
        )
        files = []
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != ".cache":
                files.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest[name] = {"repo_id": repo_id, "files": files}
    (CHECKPOINTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def download_woosh() -> None:
    destination = CHECKPOINTS / "woosh-ae"
    if (destination / "config.yaml").exists():
        return
    archive = CHECKPOINTS / "Woosh-AE.zip"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-AE.zip"
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(CHECKPOINTS)
    extracted = CHECKPOINTS / "checkpoints" / "Woosh-AE"
    if extracted.exists() and extracted != destination:
        shutil.move(str(extracted), str(destination))
    archive.unlink(missing_ok=True)


def write_provenance() -> None:
    sources: dict[str, dict[str, str]] = {}
    for name, url in REPOSITORIES.items():
        source = THIRD_PARTY / name
        commit = ""
        if (source / ".git").exists():
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
        sources[name] = {"url": url, "commit": commit}
    woosh_files = []
    woosh = CHECKPOINTS / "woosh-ae"
    if woosh.exists():
        woosh_files = [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(woosh.rglob("*"))
            if path.is_file()
        ]
    provenance = {
        "sources": sources,
        "licenses": LICENSES,
        "woosh_checkpoint_files": woosh_files,
        "environment_spec": "configs/environments.yaml",
    }
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (CHECKPOINTS / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def create_environments(names: list[str]) -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required: https://docs.astral.sh/uv/")
    config = yaml.safe_load((ROOT / "configs" / "environments.yaml").read_text())
    ENVS.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = config["environments"][name]
        environment = ENVS / name
        python = str(spec["python"])
        if not environment.exists():
            run(["uv", "venv", "--python", python, str(environment)])
        pip = ["uv", "pip", "install", "--python", str(environment / "bin" / "python")]
        source_name = {
            "mmaudio": "mmaudio",
            "same": "stable-audio-3",
            "woosh": "woosh",
        }.get(name)
        source = THIRD_PARTY / source_name if source_name else None
        if source is not None and source.is_dir():
            run(pip + ["-e", str(source)])
        # Install explicit runtime pins last because upstream editable packages may
        # otherwise upgrade Torch independently of Torchaudio.
        run(pip + list(spec["packages"]))
        run(pip + ["-e", str(ROOT), "--no-deps"])


def build_visqol() -> None:
    source = THIRD_PARTY / "visqol"
    if not source.exists():
        clone_sources(["visqol"])
    run(["bazel", "build", "-c", "opt", "//:visqol"], cwd=source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(REPOSITORIES),
        choices=list(REPOSITORIES),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=list(HF_SNAPSHOTS),
        choices=list(HF_SNAPSHOTS),
    )
    parser.add_argument(
        "--envs",
        nargs="*",
        default=["audioldm2", "mmaudio", "stable-audio-open", "same", "woosh"],
    )
    parser.add_argument("--skip-sources", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--build-visqol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_sources:
        clone_sources(args.sources)
    if not args.skip_checkpoints:
        download_hf(args.checkpoints)
        download_woosh()
    if not args.skip_envs:
        create_environments(args.envs)
    if args.build_visqol:
        build_visqol()
    write_provenance()


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
