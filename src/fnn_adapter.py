"""Loader for the licensed MICrONS foundation model used by MEDI/MESI."""
import os, sys
from pathlib import Path

def load_model(session, scan_idx, device="cuda"):
    root = Path(os.environ.get("FNN_ROOT", "../fnn"))
    package_root = root if (root / "fnn").exists() else root.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from fnn import microns
    params = root / "data/microns_digital_twin/params"
    if not params.exists():
        params = root / "model/params"
    model, units = microns.scan(session=int(session), scan_idx=int(scan_idx), directory=str(params))
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(units, "columns") and "readout_id" in units.columns:
        readout_ids = [int(x) for x in units["readout_id"]]
    else:
        readout_ids = [int(x) for x in units.index]
    return model, readout_ids
