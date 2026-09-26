"""Generate MEDIs from a population manifest.

The model loader is supplied as ``package.module:function`` and must return
``(model, model_readout_ids)`` for a session and scan.  Keeping that adapter
external avoids redistributing restricted MICrONS model weights while making
the optimization itself fully inspectable and executable.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd
import torch
from diffusers import AutoencoderTiny

from .optimizer import generate_medi_batch

REQUIRED = {"session", "scan_idx", "readout_id", "brain_area", "layer"}


def import_callable(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper MEDIs from a readout manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-loader", required=True,
                        help="Callable package.module:function returning (model, readout_ids)")
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-frames", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--chunk-overlap", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if manifest.duplicated(["session", "scan_idx", "readout_id"]).any():
        raise ValueError("manifest contains duplicate model readouts")
    loader = import_callable(args.model_loader)
    vae = AutoencoderTiny.from_pretrained(args.vae, local_files_only=True).to(args.device).eval()
    vae.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (session, scan_idx), scan_rows in manifest.groupby(["session", "scan_idx"], sort=True):
        model, model_readout_ids = loader(int(session), int(scan_idx), args.device)
        model = model.to(args.device).eval()
        model.requires_grad_(False)
        index = {int(readout_id): position for position, readout_id in enumerate(model_readout_ids)}
        records = list(scan_rows.itertuples(index=False))
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            paths = [args.output_dir / str(row.brain_area) / str(row.layer) /
                     f"{int(row.session)}_{int(row.scan_idx)}_r{int(row.readout_id)}.gif" for row in batch]
            if all(path.exists() and path.with_suffix(path.suffix + ".json").exists() for path in paths):
                continue
            videos = generate_medi_batch(
                vae, model, [index[int(row.readout_id)] for row in batch],
                total_frames=args.total_frames, chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap, iterations=args.iterations,
                lr=args.learning_rate, seed=args.seed, device=args.device,
            )
            for row, path, video in zip(batch, paths, videos):
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(path, video, fps=30, loop=0)
                metadata = {
                    "session": int(row.session), "scan_idx": int(row.scan_idx),
                    "readout_id": int(row.readout_id), "seed": args.seed,
                    "total_frames": args.total_frames, "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap, "iterations": args.iterations,
                    "learning_rate": args.learning_rate,
                }
                path.with_suffix(path.suffix + ".json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
