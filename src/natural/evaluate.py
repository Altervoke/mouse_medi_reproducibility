"""Generate natural controls from the foundation-model training arrays."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse


def best_window(values, width=60, clip=300):
    values = np.asarray(values, dtype=np.float32)
    n = len(values) // clip
    if n == 0:
        return float(values.mean()), 0, min(width, len(values)) - 1
    blocks = values[: n * clip].reshape(n, clip)
    cumulative = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(blocks, axis=1)], axis=1)
    sums = cumulative[:, width:] - cumulative[:, :-width]
    block, start = np.unravel_index(np.argmax(sums), sums.shape)
    first = int(block * clip + start)
    return float(sums[block, start] / width), first, first + width - 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate natural controls from training stimulus/response arrays")
    parser.parse_args()
    required = ["NATURAL_ROOT", "FNN_ROOT", "MICRONS_SRC"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Set licensed external artifact roots: " + ", ".join(missing))
    import sys
    import torch
    ROOT = Path(os.environ["NATURAL_ROOT"])
    FNN = Path(os.environ["FNN_ROOT"])
    SRC = Path(os.environ["MICRONS_SRC"])
    if not all(path.exists() for path in (ROOT, FNN, SRC)):
        raise RuntimeError("Configured external artifact roots must exist")
    sys.path[:0] = [str(SRC.parent), str(FNN)]
    from fnn import microns
    manifest_path = Path(os.environ.get("NATURAL_MANIFEST", str(Path(__file__).resolve().parents[2] / "data" / "baseline_manifest.csv")))
    cohort = pd.read_csv(manifest_path)
    required_columns = {"session", "scan_idx", "readout_id", "unit_id", "brain_area", "layer"}
    if not required_columns.issubset(cohort.columns):
        raise ValueError(f"natural manifest missing columns: {sorted(required_columns - set(cohort.columns))}")
    units_path = FNN / "model/params/units.csv"
    if not units_path.exists():
        units_path = ROOT / "properties" / "responses" / "units.csv"
    if not units_path.exists():
        raise FileNotFoundError(f"units.csv not found under {FNN} or {ROOT}")
    units = pd.read_csv(units_path)
    shard = int(os.environ.get("NATURAL_SHARD", "0"))
    shards = int(os.environ.get("NATURAL_SHARDS", "1"))
    cohort = cohort.iloc[shard::shards].copy()
    output_dir = Path(os.environ.get("NATURAL_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "data" / "natural_shards")))
    device = os.environ.get("NATURAL_DEVICE", "cuda")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set NATURAL_DEVICE=cpu for a smoke test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"responses_shard_{shard:02d}.csv"
    previous = pd.read_csv(output) if output.exists() else pd.DataFrame()
    done = set(zip(previous.get("session", []), previous.get("scan_idx", []), previous.get("readout_id", [])))
    records = previous.to_dict("records")
    for (session, scan), group in cohort.groupby(["session", "scan_idx"], sort=True):
        session, scan = int(session), int(scan)
        model, metadata = microns.scan(session=session, scan_idx=scan, directory=str(SRC))
        model = model.to(device).eval()
        readout_ids = metadata["readout_id"].to_numpy() if "readout_id" in metadata else metadata.index.to_numpy()
        readout_index = {int(value): i for i, value in enumerate(readout_ids)}
        response_root = ROOT / "properties" / "responses"
        response_array = response_root / "responses.npy"
        stimulus_array = response_root / "stimulus.npy"
        if response_array.exists() and stimulus_array.exists():
            observed = np.load(response_array, mmap_mode="r")
            stimuli = np.load(stimulus_array, mmap_mode="r")
            for row in group.itertuples(index=False):
                key = (session, scan, int(row.readout_id))
                if key in done:
                    continue
                unit_rows = units[(units.session == session) & (units.scan_idx == scan) & (units.unit_id == int(row.unit_id))]
                if unit_rows.empty:
                    raise ValueError(f"unit {row.unit_id} not found in units.csv for {session}/{scan}")
                unit_index = int(unit_rows.index[0])
                score, start, end = best_window(observed[unit_index])
                video = np.asarray(stimuli[start:end + 1], dtype=np.uint8)
                batch_input = np.transpose(video[None], (1, 0, 2, 3))[..., None]
                model.reset()
                with torch.no_grad():
                    predicted = np.stack(list(model.generate_response(batch_input, reset=True)), axis=0)
                if predicted.ndim == 3:
                    predicted = predicted[:, 0, :]
                value = float(predicted[10:, readout_index[int(row.readout_id)]].mean())
                records.append({"session": session, "scan_idx": scan, "unit_id": int(row.unit_id), "readout_id": int(row.readout_id), "brain_area": row.brain_area, "layer": row.layer, "start_frame": start, "end_frame": end, "natural_response": value})
                pd.DataFrame(records).to_csv(output, index=False)
            del model
            continue
        raise RuntimeError("responses.npy and stimulus.npy are required under NATURAL_ROOT/properties/responses")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
