# MESI baseline generation

`optimizer.py` implements static maximum-excitation optimization and `generate.py` provides the manifest-driven entry point. MESI optimizes a single static image with the same autoencoder/frozen-model interface as MEDI. The TAESD checkpoint is bundled at `models/taesd/`; MICrONS model parameters remain licensed external inputs.

```bash
python -m src.mesi.generate --help
python -m src.mesi.generate --output-dir data/generated_mesi \
  --manifest data/baseline_manifest.csv \
  --model-loader my_adapter:load_model --vae models/taesd
```

The output directory contains one image/JSON pair per manifest readout. The licensed model adapter and checkpoint are external inputs; JSON sidecars record identifiers and responses. Reproduction requires the same manifest, checkpoint, preprocessing, optimizer settings, and device. Published values are frozen in `data/baseline_responses.csv`.
