# Gabor baseline generation

This directory contains the production static and dynamic Gabor filters used in the manuscript. `static.py` optimizes one spatial Gabor parameterization and evaluates a 60-frame presentation; `dynamic.py` additionally optimizes temporal frequency and velocity. Both require the licensed foundation-model adapter and a MEDI GIF; model weights and MICrONS data are not redistributed.

Inspect the complete interfaces with:

```bash
python -m src.gabor.static --help
python -m src.gabor.dynamic --help
```

Each command requires `--project-root`, `--session`, `--scan`, `--readout`, `--medi`, and `--out`. The released protocol uses 60 optimization frames, 60 evaluation frames, and 30 iterations. Dynamic Gabor uses temporal-frequency and velocity bounds of 0.32 and 0.006. Outputs are a GIF plus JSON sidecar.

To reproduce a result, provide the same licensed project root, checkpoint, MEDI input, preprocessing, device, and command-line values. The public `data/baseline_responses.csv` is the frozen 800-readout result used by figures.
