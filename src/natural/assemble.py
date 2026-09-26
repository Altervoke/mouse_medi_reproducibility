"""Merge natural-control evaluator shards into the released table."""
from pathlib import Path
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a=p.parse_args()
    files=sorted(a.input_dir.glob('responses_shard_*.csv'))
    if not files: raise FileNotFoundError(f'no natural shards in {a.input_dir}')
    d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    key=['session','scan_idx','readout_id']
    if d.duplicated(key).any(): raise ValueError('natural shards contain duplicate readout keys')
    m=pd.read_csv(a.manifest)
    if set(key)-set(m.columns): raise ValueError('manifest lacks natural key columns')
    if set(map(tuple,d[key].to_numpy())) != set(map(tuple,m[key].to_numpy())):
        raise ValueError('natural shards do not exactly cover the released manifest')
    a.output.parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False); print(f'wrote {a.output} ({len(d)} rows)')
if __name__=='__main__': main()
