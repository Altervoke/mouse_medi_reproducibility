# MEDI generation

`optimizer.py` contains the latent-space optimizer used for the paper. It optimizes 60-frame grayscale stimuli in overlapping 15-frame windows (8-frame overlap), using 20 SGD updates per window, momentum 0.9, a cosine learning-rate schedule from 10 to 1, and latent noise with standard deviation 0.05.

The objective weights are 3 for negative log activation, 500 for spatial latent smoothness, 5,000 for framewise pixel smoothness, and 2,000 for framewise latent smoothness. No random spatial transform is applied.

The repository includes the TAESD checkpoint in `models/taesd/`. The MICrONS digital-twin parameters remain external licensed artifacts. After obtaining those parameters, provide a small model adapter and run:

```bash
python -m src.medi.generate \
  --manifest data/readout_manifest.csv \
  --model-loader my_adapter:load_model \
  --vae models/taesd \
  --output-dir data/generated_medis
```

The adapter signature is `load_model(session, scan_idx, device)` and its return value is `(model, readout_ids)`. The model must expose `reset()` and accept one frame plus pupil and modulation tensors, matching the public MICrONS foundation model interface.
