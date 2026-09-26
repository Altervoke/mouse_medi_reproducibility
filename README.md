# MEDI transformation-specific tolerance: reproducibility repository

This repository contains the manuscript source, 21 computational figure PDFs plus the author-drawn Figure 1 artwork (22 PDFs total), one semantic entry point per computational figure, the MEDI optimizer, and the released analysis tables used by the manuscript. Baseline-generation implementations are organized as `src/mesi/`, `src/gabor/`, `src/grating/`, and `src/natural/`; baseline-table assembly is in `src/baselines/assemble.py`.

The released readout-level tables and figure inputs are frozen outputs of the licensed 104,171-readout analysis. Together they are sufficient to regenerate every reported figure, but they are not a replacement for the licensed model-evaluation inputs. `figures/tables/` contains the compact released tables and `figures/inputs/` contains additional frozen inputs used by selected figures. `run_all.py` rebuilds derived tables and figures from those frozen inputs; it does not reconstruct the primary parameter table from the dense response matrix. Evaluator commands require licensed external model/data roots and are separate from `run_all.py`. Restricted model weights, transformation-evaluation exports, and the dense response matrix are not redistributed; required interfaces are documented in the relevant `src/*/README.md` files.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

This rebuilds the registered derived analysis tables from the released readout-level exports, renders the computational figures in `figures/pdf/`, verifies the author-drawn Figure 1 asset, and compiles `paper/main.tex`. It reads the released compact tables and figure inputs; it does not recreate the restricted dense response matrix or the primary parameter table. Each computational figure has one public entry point in `scripts/`; those scripts render figures only. `src/` contains the canonical table pipeline, the licensed evaluator interfaces, MEDI and baseline-generation implementations, and shared helpers.

### Code organization

`src/table_pipeline.py` is the canonical and only table pipeline. Its `matrix` command streams the dense response matrix and writes selectivity (including the leave-one-out table), `evaluations` converts transformation exports into readout-level retention tables, and `derived` rebuilds compact summaries and tests.

`src/tolerance.py` contains the canonical grayscale MEDI transformations and retention evaluator used by the transformation-evaluation code. Rotation fits a freely oriented ellipse to the 60-frame per-pixel range map, uses its fitted center as the rotation center, and rotates each frame with bicubic interpolation and frame-mean padding. The same module implements contrast, global-pixel translation, reversal, and variable-duration speed changes. The released `parameter_readouts.csv.gz` is a frozen compact output of that evaluation; `run_all.py` does not regenerate it without the licensed transformation-evaluation export.

For downstream visualization, `src/medi_transform.py` exposes one standalone function that transforms one grayscale GIF at a time:

```bash
python -c "from src.medi_transform import transform_gif; transform_gif('medi.gif', 'rotation', -15, 'rotation_-15.gif')"
```

The call accepts exactly the 55 canonical variants: six contrast factors, fourteen rotations, twenty-eight signed x/y translations, temporal reversal at `-1`, three speed-up factors, and three slow-down factors. Translation parameters are strings such as `x+8`, `x-16`, `y+4`, or `y-48`. Any other transformation or parameter raises `ValueError`; the function writes only the requested GIF and returns its `Path`.

The `scripts/` directory intentionally contains only `fig*.py` entry points, one per computational figure. They consume released tables and write the corresponding figure PDF; no patch or data-refresh script belongs there. `fig01_medi_pipeline.pdf` is intentionally distributed as its original author-drawn vector artwork and has no Python generator.

## Generate MEDIs

After obtaining the MICrONS digital-twin parameters and TAESD checkpoint, use `python -m src.medi.generate`. The complete command, manifest schema, model adapter contract, and production hyperparameters are documented in `src/medi/README.md`.

## Foundation-model dependency

Baseline evaluators use the public MICrONS foundation-model repository (`fnn`) and its licensed digital-twin data. Clone it beside this repository (the two directories should share the same parent):

```bash
cd /path/to/microns
git clone https://github.com/cajal/fnn.git
pip install -e fnn
```

The expected layout is `microns/fnn/` next to `microns/mouse_medi_reproducibility/`. Set the external paths required by the natural evaluator before running it:

```bash
export FNN_ROOT=/path/to/microns/fnn
export MICRONS_SRC=/path/to/microns/fnn
export NATURAL_ROOT=/path/to/microns/fnn/data/microns_digital_twin
```

`NATURAL_ROOT/properties/responses/` must contain the foundation-model training arrays `stimulus.npy` and `responses.npy`. The evaluator first selects the best contiguous 60-frame window from `responses.npy`, then re-runs the corresponding frames from `stimulus.npy` through the frozen model and averages the final 50 outputs. The arrays are large licensed inputs and are therefore not bundled in this repository.

The evaluator also accepts `NATURAL_MANIFEST`, `NATURAL_SHARD`, `NATURAL_SHARDS`, `NATURAL_OUTPUT_DIR`, and `NATURAL_DEVICE`. Gabor, grating, MESI, and MEDI commands similarly require a licensed model adapter/checkpoint; their exact interfaces are documented in each `src/*/README.md`. The frozen tables and figures can be regenerated without those external artifacts.

## Rebuilding derived tables

The canonical table entry point is `src/table_pipeline.py`. It keeps the large-input steps explicit and produces all files in `figures/tables/`:

```bash
# From a dense MEDI response matrix and matching sorted manifest:
python -m src.table_pipeline matrix --matrix /data/medi_responses.npy \
  --manifest /data/readout_manifest.csv

# From a transformation-evaluation export and the matrix-derived selectivity:
python -m src.table_pipeline evaluations \
  --evaluations /data/transformation_evaluations.csv.gz

# Rebuild compact summaries, tests, and mechanism tables:
python -m src.table_pipeline derived
python -m src.table_pipeline verify
```

The `matrix` command streams the square response matrix and writes the selection, layer, area-summary, and within-readout block tables. The `evaluations` command expects one row per tested transformation and requires `session`, `scan_idx`, `readout_id`, `brain_area`, `condition`, `parameter`, `response`, and `original_response`; it writes the parameter-level and neuron-family tables. The `derived` command consumes the released readout-level tables and deterministically rebuilds every compact statistical table, including baseline tests, quality subsets, nested splits, and reverse mechanism summaries. `verify` fails if a table is missing or an unregistered CSV has appeared in `figures/tables/`.

The canonical `table_pipeline.py` commands above are the only supported table-generation interface. Dense response-matrix summaries must be produced by the licensed block evaluator before their compact tables can be regenerated; no large matrix is silently substituted.
