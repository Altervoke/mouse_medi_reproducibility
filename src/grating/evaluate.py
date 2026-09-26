"""Evaluate the production 480-condition drifting-grating grid."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from .stimuli import condition_grid, render_grating


def evaluate_grid(model, readout_indices, device="cuda"):
    """Return response values with shape ``(readouts, 480)``.

    The adapter must implement ``generate_response(frames, reset=True)``.
    ``frames`` has shape ``(time, conditions, height, width, channels)`` and
    the returned iterable yields one ``(conditions, readouts)`` array per time
    step. The first ten steps are discarded before averaging.
    """
    import torch

    del device
    conditions = list(condition_grid())
    rendered = np.stack([
        np.rint(render_grating(c["direction"], c["temporal_frequency"], c["spatial_frequency"]) * 255).astype(np.uint8)
        for c in conditions
    ])
    batch = np.transpose(rendered, (1, 0, 2, 3))[..., None]
    with torch.no_grad():
        trajectory = np.stack(list(model.generate_response(batch, reset=True)), axis=0)
    if trajectory.ndim != 3:
        raise ValueError(f"adapter responses must stack to (time, conditions, readouts), got {trajectory.shape}")
    values = trajectory[10:].mean(0)[:, np.asarray(readout_indices)]
    return values.T


class MockAdapter:
    """Deterministic adapter for interface and shape smoke tests."""

    def generate_response(self, frames, reset=True):
        del reset
        n_conditions = frames.shape[1]
        base = np.arange(n_conditions, dtype=np.float32)[:, None]
        return [np.repeat(base, 2, axis=1) + float(t) / 100.0 for t in range(frames.shape[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="run a fast adapter smoke test")
    parser.add_argument("--readouts", type=int, default=1, help="number of mock readouts")
    parser.add_argument("--output", type=Path, help="optional .npy output for mock responses")
    args = parser.parse_args()
    if not args.mock:
        parser.error("the CLI provides --mock for interface validation; call evaluate_grid(...) with a licensed model adapter for production evaluation")
    if args.readouts < 1:
        parser.error("--readouts must be positive")
    n_conditions = len(list(condition_grid()))
    frames = np.zeros((60, n_conditions, 1, 1, 1), dtype=np.uint8)
    trajectory = np.stack(list(MockAdapter().generate_response(frames, reset=True)), axis=0)
    values = trajectory[10:].mean(0)[:, list(range(args.readouts))].T
    if values.shape != (args.readouts, 480):
        raise RuntimeError(f"mock output shape mismatch: {values.shape}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, values)
    print(f"mock drifting-grating evaluation passed: shape={values.shape}, conditions={n_conditions}")


if __name__ == "__main__":
    main()
