# Drifting-grating baseline

`stimuli.py` defines the complete 480-condition grid: 16 directions uniformly spaced over 360 degrees, 5 temporal frequencies from 0.5--8 Hz, and 6 spatial frequencies from 0.01--0.32 cycles per pixel. `evaluate.py` renders and scores the grid with a supplied frozen-model adapter, discarding the first 10 frames and returning a `(readouts, 480)` response matrix. The maximum over the 480 columns is the released drifting-grating response.

For a licensed model, call `evaluate_grid(model, readout_indices)` from Python. The adapter must implement `generate_response(frames, reset=True)`, where `frames` has shape `(time, conditions, height, width, channels)` and the returned iterable yields one `(batch, readouts)` response array per time step. The evaluator does not require a separate `reset()` method; `reset=True` is passed to the model call so the recurrent state is reset for the grid.

Verify the public interface without model weights with:

```bash
python -m src.grating.evaluate --mock --readouts 2 --output /tmp/grating_mock.npy
```

This deterministic smoke test checks the 480-condition rendering, adapter call shape, warm-up averaging, and output shape. Model weights and a production adapter remain licensed external inputs. The frozen 800-readout outputs are in `data/baseline_responses.csv`; `src.baselines.assemble` validates their schema and exact readout alignment.