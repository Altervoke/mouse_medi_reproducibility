"""Production drifting-grating generator and response-table builder."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
DIRECTIONS=np.linspace(0.0,2*np.pi,16,endpoint=False)
TEMPORAL_FREQUENCIES=np.geomspace(0.5,8.0,5)
SPATIAL_FREQUENCIES=np.geomspace(0.01,0.32,6)
def render_grating(direction, temporal_frequency, spatial_frequency, *, frames=60, height=144, width=256, frame_rate=30.0):
 y,x=np.mgrid[0:height,0:width].astype(np.float32); x=x-(width-1)/2; y=y-(height-1)/2
 coordinate=x*np.cos(direction)+y*np.sin(direction); t=np.arange(frames,dtype=np.float32)[:,None,None]/frame_rate
 return (0.5+0.5*np.sin(2*np.pi*(spatial_frequency*coordinate[None]+temporal_frequency*t))).astype(np.float32)
def condition_grid():
 for di,d in enumerate(DIRECTIONS):
  for ti,tf in enumerate(TEMPORAL_FREQUENCIES):
   for si,sf in enumerate(SPATIAL_FREQUENCIES): yield {'direction_index':di,'tf_index':ti,'sf_index':si,'direction':d,'temporal_frequency':tf,'spatial_frequency':sf}
def select_preferred(responses):
 values=np.asarray(responses); conditions=list(condition_grid())
 if values.ndim!=2 or values.shape[1]!=len(conditions): raise ValueError('responses must have shape [readout, 480]')
 best=np.argmax(values,axis=1); rows=[]
 for row,index in enumerate(best):
  item={k:v for k,v in conditions[int(index)].items() if k!='direction'}; item.update(condition_index=int(index),response=float(values[row,index])); rows.append(item)
 return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--responses',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); a=p.parse_args()
    if a.responses.suffix == '.npy':
        values = np.load(a.responses)
    else:
        raw = pd.read_csv(a.responses)
        metadata = {'session', 'scan_idx', 'unit_id', 'readout_id', 'brain_area', 'layer'}
        response_columns = [c for c in raw.columns if c not in metadata]
        if len(response_columns) != 480:
            raise ValueError(f'expected 480 response columns, found {len(response_columns)}')
        values = raw[response_columns].to_numpy(dtype=np.float32)
    preferred=select_preferred(values); manifest=pd.read_csv(a.manifest)
    if len(manifest)!=len(preferred): raise ValueError('manifest and response matrix row counts differ')
    manifest=manifest.reset_index(drop=True)
    if manifest[['session','scan_idx','unit_id','readout_id']].duplicated().any():
        raise ValueError('manifest contains duplicate readout identifiers')
    if a.responses.suffix != '.npy':
        response_keys = raw[['session','scan_idx','unit_id','readout_id']].reset_index(drop=True)
        manifest_keys = manifest[['session','scan_idx','unit_id','readout_id']]
        if not response_keys.equals(manifest_keys):
            raise ValueError('response CSV key order does not exactly match the manifest')
    out=pd.concat([manifest[['session','scan_idx','unit_id','readout_id','brain_area','layer']],preferred],axis=1)
    a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f'wrote {a.output} ({len(out)} readouts; 480 conditions)')
if __name__=='__main__': main()
