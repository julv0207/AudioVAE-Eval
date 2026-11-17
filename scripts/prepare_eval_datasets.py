#!/usr/bin/env python3
"""Analyze and materialize the AudioCaps and NC-AI evaluation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SAMPLE_RATE = 44_100
DURATION_SECONDS = 10
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
N_FFT = 2048
HOP_LENGTH = 1024
ACTIVE_FLOOR_DBFS = -80.0


def resolve_nc_audio(root: Path, item: dict[str, Any]) -> Path:
    stem = Path(item["filename"]).stem
    prefix = f"{int(item['era']):07d}_{stem}"
    category = str(item["category"])
    for extension in (".flac", ".wav", ".ogg", ".mp3"):
        candidate = root / "SFX" / category / f"{prefix}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"NC-AI audio not found for {prefix}")


def load_ten_seconds(path: Path) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    source_rate = int(info.samplerate)
    source_target_samples = source_rate * DURATION_SECONDS
    audio, sample_rate = sf.read(
        path,
        frames=source_target_samples,
        dtype="float32",
        always_2d=True,
    )
    valid_seconds = audio.shape[0] / sample_rate
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    valid_samples = min(round(valid_seconds * SAMPLE_RATE), TARGET_SAMPLES)
    if audio.shape[0] < TARGET_SAMPLES:
        audio = np.pad(audio, ((0, TARGET_SAMPLES - audio.shape[0]), (0, 0)))
    else:
        audio = audio[:TARGET_SAMPLES]
    return audio, valid_samples


def high_frequency_features(audio: np.ndarray, cutoff_hz: float) -> dict[str, float]:
    mono = torch.from_numpy(audio.mean(axis=1))
    frames = mono.unfold(0, N_FFT, HOP_LENGTH)
    window = torch.hann_window(N_FFT)
    spectrum = torch.fft.rfft(frames * window, dim=-1)
    power = spectrum.abs().square()
    weights = torch.ones(power.shape[-1])
    weights[1:-1] = 2
    power = power * weights
    frequencies = torch.fft.rfftfreq(N_FFT, d=1 / SAMPLE_RATE)
    band = frequencies > cutoff_hz
    normalization = N_FFT * window.square().sum()
    total_frame_energy = power.sum(dim=-1) / normalization
    high_frame_energy = power[:, band].sum(dim=-1) / normalization
    total_energy = total_frame_energy.mean()
    high_energy = high_frame_energy.mean()
    ratio = high_energy / total_energy.clamp_min(1e-12)
    high_dbfs = 10 * torch.log10(high_energy.clamp_min(1e-12))
    frame_dbfs = 10 * torch.log10(high_frame_energy.clamp_min(1e-12))
    active_ratio = (frame_dbfs >= ACTIVE_FLOOR_DBFS).float().mean()
    return {
        "energy_ratio": float(ratio),
        "absolute_energy_dbfs": float(high_dbfs),
        "active_time_ratio": float(active_ratio),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_audiocaps_manifest(metadata_path: Path, root: Path, output: Path) -> None:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))["data"]
    dataset_root = output / "audiocaps_test"
    clip_root = dataset_root / "clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    records_by_source: dict[str, dict[str, Any]] = {}
    missing = []
    for index, item in enumerate(data, start=1):
        path = root / item["wav"]
        if not path.exists():
            missing.append(str(path))
            continue
        destination = clip_root / f"{path.stem}.wav"
        valid_destination = False
        if destination.exists():
            info = sf.info(destination)
            valid_destination = (
                info.samplerate == SAMPLE_RATE and info.frames == TARGET_SAMPLES
            )
        if not valid_destination:
            audio, _ = load_ten_seconds(path)
            channels = min(audio.shape[1], 2)
            sf.write(
                destination,
                audio[:, :channels],
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        key = str(path.resolve())
        if key not in records_by_source:
            records_by_source[key] = {
                "path": str(destination.resolve()),
                "source_path": key,
                "captions": [],
                "labels": [],
                "dataset": "audiocaps_test",
            }
        record = records_by_source[key]
        if item.get("caption") not in record["captions"]:
            record["captions"].append(item.get("caption"))
        if item.get("labels") not in record["labels"]:
            record["labels"].append(item.get("labels"))
        if index % 250 == 0:
            print(f"materialized AudioCaps {index}/{len(data)}", flush=True)
    records = list(records_by_source.values())
    write_jsonl(dataset_root / "manifest.jsonl", records)
    summary = {
        "metadata": str(metadata_path),
        "metadata_entries": len(data),
        "unique_audio": len(records),
        "missing": len(missing),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "short_clip_policy": "zero-pad at end",
        "long_clip_policy": "crop after first 10 seconds",
    }
    (dataset_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def analyze_nc(
    metadata_path: Path,
    root: Path,
    analysis_path: Path,
) -> list[dict[str, Any]]:
    if analysis_path.exists() and analysis_path.stat().st_size > 0:
        with analysis_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in metadata
        if item["split"] == "test" and int(item["sampleRate"]) == SAMPLE_RATE
    ]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        path = resolve_nc_audio(root, item)
        audio, valid_samples = load_ten_seconds(path)
        row: dict[str, Any] = {
            "era": int(item["era"]),
            "path": str(path.resolve()),
            "filename": item["filename"],
            "category": item["category"],
            "subCategory": item["subCategory"],
            "description": item["description"],
            "source_duration": float(item["duration"]),
            "valid_seconds": valid_samples / SAMPLE_RATE,
        }
        for cutoff in (8_000, 16_000):
            features = high_frequency_features(audio, cutoff)
            suffix = "8k" if cutoff == 8_000 else "16k"
            row.update({f"{key}_{suffix}": value for key, value in features.items()})
        rows.append(row)
        if index % 100 == 0:
            print(f"analyzed {index}/{len(candidates)}", flush=True)
    print(f"analyzed metadata-{SAMPLE_RATE} candidates: {len(rows)}", flush=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = []
    for row in rows:
        converted = dict(row)
        for cutoff in ("8k", "16k"):
            for metric in ("energy_ratio", "absolute_energy_dbfs", "active_time_ratio"):
                converted[f"{metric}_{cutoff}"] = float(converted[f"{metric}_{cutoff}"])
        converted["era"] = int(converted["era"])
        converted["source_duration"] = float(converted["source_duration"])
        converted["valid_seconds"] = float(converted["valid_seconds"])
        numeric.append(converted)
    return numeric


def distribution_summary(rows: list[dict[str, Any]], output: Path) -> None:
    summary: dict[str, Any] = {
        "count": len(rows),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "active_frame_floor_dbfs": ACTIVE_FLOOR_DBFS,
        "metrics": {},
    }
    quantiles = (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    for cutoff in ("8k", "16k"):
        summary["metrics"][cutoff] = {}
        for metric in ("energy_ratio", "absolute_energy_dbfs", "active_time_ratio"):
            values = np.asarray([row[f"{metric}_{cutoff}"] for row in rows])
            summary["metrics"][cutoff][metric] = {
                str(quantile): float(np.quantile(values, quantile))
                for quantile in quantiles
            }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def select_rows(
    rows: list[dict[str, Any]],
    cutoff: str,
    thresholds: dict[str, float],
    limit: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row[f"energy_ratio_{cutoff}"] >= thresholds["energy_ratio"]
        and row[f"absolute_energy_dbfs_{cutoff}"] >= thresholds["absolute_energy_dbfs"]
        and row[f"active_time_ratio_{cutoff}"] >= thresholds["active_time_ratio"]
    ]
    if not eligible:
        return []
    metric_names = ("energy_ratio", "absolute_energy_dbfs", "active_time_ratio")
    rank_matrix = np.stack(
        [
            percentile_ranks(
                np.asarray([row[f"{metric}_{cutoff}"] for row in eligible])
            )
            for metric in metric_names
        ],
        axis=1,
    )
    # The minimum percentile favors clips that are strong on all three criteria,
    # rather than clips dominated by only one extreme value.
    scores = rank_matrix.min(axis=1)
    ranked = sorted(
        zip(eligible, scores, strict=True),
        key=lambda pair: (-pair[1], int(pair[0]["era"])),
    )
    selected = []
    for row, score in ranked[:limit]:
        item = dict(row)
        item["richness_score"] = float(score)
        selected.append(item)
    return selected


def materialize(
    selected_by_cutoff: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    dataset_root = output / "nc_ai_hf_rich"
    clip_root = dataset_root / "clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    all_rows = {
        int(row["era"]): row
        for selected in selected_by_cutoff.values()
        for row in selected
    }
    selected_clip_names = {f"{era:07d}.wav" for era in all_rows}
    for stale in clip_root.glob("*.wav"):
        if stale.name not in selected_clip_names:
            stale.unlink()
    for index, (era, row) in enumerate(sorted(all_rows.items()), start=1):
        destination = clip_root / f"{era:07d}.wav"
        valid_destination = False
        if destination.exists():
            info = sf.info(destination)
            valid_destination = (
                info.samplerate == SAMPLE_RATE and info.frames == TARGET_SAMPLES
            )
        if not valid_destination:
            audio, _ = load_ten_seconds(Path(row["path"]))
            channels = min(audio.shape[1], 2)
            sf.write(
                destination,
                audio[:, :channels],
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        if index % 100 == 0:
            print(f"materialized {index}/{len(all_rows)}", flush=True)

    for cutoff, selected in selected_by_cutoff.items():
        split_root = dataset_root / f"hf{cutoff}"
        split_root.mkdir(parents=True, exist_ok=True)
        selected_names = {f"{int(row['era']):07d}.wav" for row in selected}
        for stale in split_root.glob("*.wav"):
            if stale.name not in selected_names:
                stale.unlink()
        manifest = []
        for row in selected:
            source = clip_root / f"{int(row['era']):07d}.wav"
            link = split_root / source.name
            if not link.exists():
                os.link(source, link)
            manifest.append(
                {
                    "path": str(link.resolve()),
                    "source_path": row["path"],
                    "era": int(row["era"]),
                    "description": row["description"],
                    "category": row["category"],
                    "subCategory": row["subCategory"],
                    "energy_ratio": row[f"energy_ratio_{cutoff}"],
                    "absolute_energy_dbfs": row[f"absolute_energy_dbfs_{cutoff}"],
                    "active_time_ratio": row[f"active_time_ratio_{cutoff}"],
                    "richness_score": row["richness_score"],
                }
            )
        write_jsonl(split_root / "manifest.jsonl", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audiocaps-metadata",
        type=Path,
        default=Path(
            "/workspace/AudioCaps/metadata/audiocaps/datafiles/audiocaps_test_label.json"
        ),
    )
    parser.add_argument(
        "--audiocaps-root", type=Path, default=Path("/workspace/AudioCaps/audioset")
    )
    parser.add_argument(
        "--nc-metadata",
        type=Path,
        default=Path("/workspace/NC_AI_data/prosoundeffect_split_info.json"),
    )
    parser.add_argument("--nc-root", type=Path, default=Path("/workspace/NC_AI_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/AudioVAE_eval/datasets"),
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_audiocaps_manifest(
        args.audiocaps_metadata,
        args.audiocaps_root,
        args.output,
    )
    analysis_path = args.output / "nc_ai_hf_rich" / "analysis_44100_test.csv"
    rows = numeric_rows(analyze_nc(args.nc_metadata, args.nc_root, analysis_path))
    distribution_summary(
        rows,
        args.output / "nc_ai_hf_rich" / "distribution_summary.json",
    )
    if args.analyze_only:
        return
    if args.thresholds is None:
        raise ValueError("--thresholds is required unless --analyze-only is used")
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    selected = {
        cutoff: select_rows(rows, cutoff, thresholds[cutoff], args.limit)
        for cutoff in ("8k", "16k")
    }
    for cutoff, items in selected.items():
        print(f"selected hf{cutoff}: {len(items)}")
        if len(items) < args.limit:
            raise RuntimeError(
                f"Only {len(items)} clips satisfy hf{cutoff} thresholds; requested {args.limit}"
            )
    materialize(selected, args.output)
    selection_summary = {
        "thresholds": thresholds,
        "selected": {key: len(value) for key, value in selected.items()},
        "unique_clips": len(
            {int(row["era"]) for values in selected.values() for row in values}
        ),
    }
    (args.output / "nc_ai_hf_rich" / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
