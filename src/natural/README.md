# Natural-movie baseline

`evaluate.py` computes the natural control from the foundation-model training `stimulus.npy` and `responses.npy` arrays. For each readout and each independent 300-frame clip it scans contiguous 60-frame windows in `responses.npy`, selects the highest-response window, re-runs those frames through the frozen model, and summarizes the final 50 outputs after the initial transient. No trial-directory fallback is used. `assemble.py` validates the released table.

```bash
python -m src.natural.evaluate --help
NATURAL_ROOT=/path/to/fnn/data/microns_digital_twin \
FNN_ROOT=/path/to/fnn MICRONS_SRC=/path/to/microns_source \
python -m src.natural.evaluate
```

The environment variables identify licensed external artifacts; optional `NATURAL_MANIFEST`, `NATURAL_SHARD(S)`, `NATURAL_OUTPUT_DIR`, and `NATURAL_DEVICE` control execution. Reproduction requires the same arrays, checkpoint, preprocessing, manifest, and device. Published values are frozen in `data/baseline_responses.csv`.
