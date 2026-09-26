"""Manifest-driven MESI image generation."""
import argparse, importlib, json
from pathlib import Path
import imageio.v2 as imageio
import pandas as pd
import torch
from diffusers import AutoencoderTiny
from .optimizer import generate_mesi_batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--model-loader',required=True); p.add_argument('--vae',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--iterations',type=int,default=10); p.add_argument('--learning-rate',type=float,default=10.0); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    mod,name=a.model_loader.split(':',1); loader=getattr(importlib.import_module(mod),name); m=pd.read_csv(a.manifest)
    vae=AutoencoderTiny.from_pretrained(a.vae,local_files_only=True).to(a.device).eval(); vae.requires_grad_(False)
    for (s,sc), group in m.groupby(['session','scan_idx']):
        model,ids=loader(int(s),int(sc),a.device); model=model.to(a.device).eval(); model.requires_grad_(False); index={int(v):i for i,v in enumerate(ids)}; rows=list(group.itertuples(index=False))
        images=generate_mesi_batch(vae,model,[index[int(r.readout_id)] for r in rows],iterations=a.iterations,lr=a.learning_rate,seed=a.seed,device=a.device)
        for r,img in zip(rows,images):
            out=a.output_dir/str(r.brain_area)/str(r.layer)/f'{int(r.session)}_{int(r.scan_idx)}_r{int(r.readout_id)}.png'; out.parent.mkdir(parents=True,exist_ok=True); imageio.imwrite(out,img); out.with_suffix('.png.json').write_text(json.dumps({'session':int(r.session),'scan_idx':int(r.scan_idx),'readout_id':int(r.readout_id),'seed':a.seed,'iterations':a.iterations,'learning_rate':a.learning_rate},indent=2)+'\n')
        del model; torch.cuda.empty_cache()
if __name__=='__main__': main()
