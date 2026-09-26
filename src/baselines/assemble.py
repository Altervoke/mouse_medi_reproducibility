"""Validate alignment and completeness of the released baseline table."""
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST_KEY_MAP = {"session": "session", "scan_idx": "scan", "readout_id": "readout"}
RESPONSE_KEYS = ["session", "scan", "readout"]
REQUIRED_VALUES = ["medi_response", "dynamic_gabor_response", "static_gabor_response", "grating_response", "MESI", "Natural"]


def _key_frame(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    missing = sorted(set(mapping) - set(frame.columns))
    if missing:
        raise ValueError(f"missing key columns: {missing}")
    return frame[list(mapping)].rename(columns=mapping).astype({k: str for k in mapping.values()})


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def main() -> None:
    manifest = pd.read_csv(DATA / "baseline_manifest.csv")
    responses = pd.read_csv(DATA / "baseline_responses.csv")
    manifest_keys = _key_frame(manifest, MANIFEST_KEY_MAP)
    response_keys = _key_frame(responses, {k: k for k in RESPONSE_KEYS})
    missing = sorted(set(REQUIRED_VALUES) - set(responses.columns))
    if missing:
        raise ValueError(f"baseline_responses.csv is missing columns: {missing}")
    if response_keys.duplicated().any():
        raise ValueError("baseline_responses.csv contains duplicate readout keys")
    if manifest_keys.duplicated().any():
        raise ValueError("baseline_manifest.csv contains duplicate readout keys")
    if len(responses) != len(manifest):
        raise ValueError(f"response rows ({len(responses)}) != manifest rows ({len(manifest)})")
    if not response_keys.equals(manifest_keys):
        missing_keys = manifest_keys.merge(response_keys, how="left", indicator=True).query("_merge == 'left_only'").shape[0]
        extra_keys = response_keys.merge(manifest_keys, how="left", indicator=True).query("_merge == 'left_only'").shape[0]
        raise ValueError(f"baseline key alignment failed: missing={missing_keys}, extra={extra_keys}, order or values differ")
    if "brain_area" in responses and "brain_area" in manifest:
        if not responses.brain_area.astype(str).reset_index(drop=True).equals(manifest.brain_area.astype(str).reset_index(drop=True)):
            raise ValueError("baseline brain_area values are not aligned with manifest")
    nulls = responses[REQUIRED_VALUES].isna().sum()
    if nulls.any():
        raise ValueError(f"missing baseline values: {nulls[nulls > 0].to_dict()}")
    print(f"validated {DATA / 'baseline_responses.csv'} ({len(responses)} rows)")
    print(f"manifest_key_sha256={_digest(manifest_keys)}")
    print(f"response_key_sha256={_digest(response_keys)}")


if __name__ == "__main__":
    main()