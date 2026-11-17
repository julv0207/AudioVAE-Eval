from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from audio_vae_eval.adapters import reconstruct
from audio_vae_eval.audio import load_audio
from audio_vae_eval.metrics import evaluate_pair
from audio_vae_eval.registry import DEFAULT_MODELS, MODEL_REGISTRY, get_model_spec

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def discover_audio(path: Path, extensions: list[str]) -> list[Path]:
    extensions = {extension.lower() for extension in extensions}
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in extensions
        )
    if path.suffix.lower() in {".json", ".jsonl"}:
        content = path.read_text(encoding="utf-8")
        entries = (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if path.suffix.lower() == ".jsonl"
            else json.loads(content)
        )
        if isinstance(entries, dict):
            entries = entries.get("files", [])
        return [
            (path.parent / (entry["path"] if isinstance(entry, dict) else entry)).resolve()
            for entry in entries
        ]
    if path.suffix.lower() not in extensions:
        raise ValueError(f"Unsupported audio extension: {path}")
    return [path.resolve()]


def command_bootstrap(args: argparse.Namespace) -> int:
    command = [sys.executable, str(ROOT / "scripts" / "bootstrap.py")]
    for option in ("sources", "checkpoints", "envs"):
        values = getattr(args, option)
        if values is not None:
            command.extend([f"--{option}", *values])
    for option in (
        "skip_sources",
        "skip_checkpoints",
        "skip_envs",
        "build_visqol",
    ):
        if getattr(args, option):
            command.append(f"--{option.replace('_', '-')}")
    return subprocess.run(command).returncode


def command_list_models(_: argparse.Namespace) -> int:
    for name, spec in MODEL_REGISTRY.items():
        print(
            json.dumps(
                {
                    **spec.to_dict(),
                    "environment_ready": (
                        ROOT / ".venvs" / spec.environment / "bin" / "python"
                    ).exists(),
                }
            )
        )
    return 0


def requested_models(values: list[str] | None) -> list[str]:
    models = list(values or DEFAULT_MODELS)
    for model in models:
        get_model_spec(model)
    return models


def artifact_id(source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode(), usedforsecurity=False).hexdigest()[:10]
    return f"{source.stem}-{digest}"


def command_reconstruct(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    files = discover_audio(args.input, config["audio"]["extensions"])
    failures = 0
    for model in requested_models(args.models):
        for source in files:
            destination = args.output / "reconstructions" / model / artifact_id(source)
            try:
                result = reconstruct(model, source, destination, args.device, args.dtype)
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "model": model,
                            "input": str(source),
                            "output": str(result.reconstruction_path),
                        }
                    )
                )
            except Exception as error:
                failures += 1
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "model": model,
                            "input": str(source),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    ),
                    file=sys.stderr,
                )
    return int(failures > 0)


def _evaluate_one(
    model: str,
    source: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    item_id = artifact_id(source)
    result_path = args.output / "results" / model / f"{item_id}.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"model": model, "input": str(source), "status": "error"}
    try:
        reconstruction_dir = args.output / "reconstructions" / model / item_id
        reconstruction = reconstruct(
            model, source, reconstruction_dir, args.device, args.dtype
        )
        spec = get_model_spec(model)
        prepared_reference = reconstruction.metadata["prepared_reference_path"]
        reference_audio, _ = load_audio(prepared_reference, spec.sample_rate, spec.channels)
        reconstructed_audio, _ = load_audio(
            reconstruction.reconstruction_path, spec.sample_rate, spec.channels
        )
        metric_values = evaluate_pair(
            reference_audio,
            reconstructed_audio,
            spec.sample_rate,
            config,
            project_root=ROOT,
        )
        ceiling_metrics = None
        ceiling_path = reconstruction.metadata.get("vocoder_ceiling_path")
        if ceiling_path:
            ceiling_audio, _ = load_audio(ceiling_path, spec.sample_rate, spec.channels)
            ceiling_metrics = evaluate_pair(
                reference_audio,
                ceiling_audio,
                spec.sample_rate,
                config,
                project_root=ROOT,
            )
        record.update(
            {
                "status": "ok",
                "reconstruction": str(reconstruction.reconstruction_path),
                "metadata": reconstruction.metadata,
                "metrics": metric_values,
                "vocoder_ceiling_metrics": ceiling_metrics,
            }
        )
    except Exception as error:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.no_visqol:
        config["metrics"]["visqol"]["enabled"] = False
    if args.hf_cutoffs:
        config["metrics"]["spectral"]["hf_cutoffs_hz"] = args.hf_cutoffs
    files = discover_audio(args.input, config["audio"]["extensions"])
    records = []
    for model in requested_models(args.models or config.get("models")):
        for source in files:
            record = _evaluate_one(model, source, args, config)
            records.append(record)
            print(json.dumps({key: record.get(key) for key in ("status", "model", "input", "error")}))
    jsonl = args.output / "results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return int(any(record["status"] != "ok" for record in records))


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = sorted(path.glob("*/*.json"))
        return [json.loads(candidate.read_text(encoding="utf-8")) for candidate in candidates]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def command_report(args: argparse.Namespace) -> int:
    records = [record for record in _load_records(args.results) if record["status"] == "ok"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        row = {
            "model": record["model"],
            "input": record["input"],
            "sample_rate": metadata.get("sample_rate"),
            "temporal_stride": metadata.get("temporal_stride"),
            "latent_frame_rate": metadata.get("latent_frame_rate"),
            "vae_parameters": metadata.get("vae_parameters"),
            "encoder_parameters": metadata.get("encoder_parameters"),
            "decoder_parameters": metadata.get("decoder_parameters"),
            "vocoder_parameters": metadata.get("vocoder_parameters"),
            **record["metrics"],
        }
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["model", "input"]
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        aggregate[model] = {}
        for key in fieldnames:
            values = [
                float(row[key])
                for row in model_rows
                if isinstance(row.get(key), (int, float)) and row.get(key) is not None
            ]
            if values:
                aggregate[model][key] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values),
                    "count": len(values),
                }
    args.output.with_suffix(".json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.json')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-vae-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--sources", nargs="*")
    bootstrap.add_argument("--checkpoints", nargs="*")
    bootstrap.add_argument("--envs", nargs="*")
    bootstrap.add_argument("--skip-sources", action="store_true")
    bootstrap.add_argument("--skip-checkpoints", action="store_true")
    bootstrap.add_argument("--skip-envs", action="store_true")
    bootstrap.add_argument("--build-visqol", action="store_true")
    bootstrap.set_defaults(function=command_bootstrap)
    list_models = subparsers.add_parser("list-models")
    list_models.set_defaults(function=command_list_models)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--models", nargs="+")
        command.add_argument("--output", type=Path, default=ROOT / "outputs")
        command.add_argument("--config", type=Path, default=ROOT / "configs" / "eval.yaml")
        command.add_argument("--device", default="cuda")
        command.add_argument("--dtype", default="float32")

    reconstruct_parser = subparsers.add_parser("reconstruct")
    add_run_arguments(reconstruct_parser)
    reconstruct_parser.set_defaults(function=command_reconstruct)
    evaluate = subparsers.add_parser("evaluate")
    add_run_arguments(evaluate)
    evaluate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--no-visqol", action="store_true")
    evaluate.add_argument(
        "--hf-cutoffs",
        type=float,
        nargs="+",
        help="STFT-HF cutoff frequencies in Hz (for example: 8000 or 16000)",
    )
    evaluate.set_defaults(function=command_evaluate)
    report = subparsers.add_parser("report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path, default=ROOT / "outputs" / "summary")
    report.set_defaults(function=command_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
