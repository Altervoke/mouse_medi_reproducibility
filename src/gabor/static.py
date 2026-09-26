"""Static-Gabor baseline using the production dynamic-Gabor optimizer.

The implementation is intentionally identical to the dynamic optimizer except
that temporal frequency and velocity are fixed at zero (and excluded from the
optimizer parameter list).
"""
import argparse, json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import torch
from PIL import Image, ImageDraw, ImageFont
try:
    from .dynamic import load_model, load_gif, response, FitGaborDynamic
except ImportError:  # pragma: no cover - direct script execution
    from dynamic import load_model, load_gif, response, FitGaborDynamic

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True); p.add_argument('--session',type=int,required=True); p.add_argument('--scan',type=int,required=True); p.add_argument('--readout',type=int,required=True); p.add_argument('--medi',required=True); p.add_argument('--out',required=True); p.add_argument('--iterations',type=int,default=30); p.add_argument('--frames',type=int,default=60); p.add_argument('--opt-frames',type=int,default=60); p.add_argument('--device',default='cuda'); a=p.parse_args()
    if a.opt_frames != a.frames:
        raise ValueError('static-Gabor optimization and evaluation must both use 60 frames')
    dev=a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu'; model,units=load_model(a.project_root,a.session,a.scan,dev)
    ids=[int(x) for x in units['readout_id'].tolist()] if 'readout_id' in units.columns else [int(x) for x in units.index.tolist()]; idx=ids.index(a.readout)
    medi=load_gif(a.medi,a.frames)
    if medi.shape[1:] != (144, 256):
        import cv2
        medi=np.stack([cv2.resize(x,(256,144),interpolation=cv2.INTER_AREA) for x in medi])
    medi_r=response(model,medi,idx,dev); torch.manual_seed(7000+a.readout); g=FitGaborDynamic(frames=a.opt_frames,device=dev,temporal_frequency_bound=0.0,velocity_bound=0.0)
    with torch.no_grad(): g.temporal_frequency.zero_(); g.velocity.zero_()
    g.temporal_frequency.requires_grad_(False); g.velocity.requires_grad_(False); opt=torch.optim.Adam([x for x in g.parameters() if x.requires_grad],lr=.05); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5); best=(-1e9,None)
    for i in range(a.iterations):
        opt.zero_grad(); v=g(); model.reset(); x=v[:,None]; z=torch.zeros((len(v),2),device=dev); y=torch.cat([model(x[t:t+1],z[t:t+1],z[t:t+1]) for t in range(len(v))]); loss=-y[10:,idx].mean(); score=-float(loss.detach().cpu())
        if score>best[0]: best=(score,{k:q.detach().clone() for k,q in g.named_parameters()})
        loss.backward(); opt.step(); g.project_(); sched.step(score); print(f'iter {i+1}/{a.iterations} response={score:.4f}',flush=True)
    with torch.no_grad():
        for k,q in best[1].items(): getattr(g,k).copy_(q)
        g.frames=a.frames; gab=g().cpu().numpy()
    gab_r=response(model,gab,idx,dev); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); panels=[]
    try: font=ImageFont.truetype('DejaVuSans-Bold.ttf',18)
    except Exception: font=ImageFont.load_default()
    for t in range(a.frames):
        l=(medi[t]*255).astype('uint8'); r=(gab[t]*255).astype('uint8'); c=np.concatenate([l,np.full((l.shape[0],8),240,'uint8'),r],1); q=np.full((c.shape[0]+30,c.shape[1]),245,'uint8'); q[30:]=c; im=Image.fromarray(q); d=ImageDraw.Draw(im); d.text((l.shape[1]//2-25,6),'MEDI',fill=20,font=font); d.text((l.shape[1]+8+r.shape[1]//2-60,6),'static Gabor',fill=20,font=font); panels.append(np.asarray(im))
    imageio.mimsave(out,panels,fps=30,loop=0); meta={'session':a.session,'scan':a.scan,'readout':a.readout,'medi_response':medi_r,'static_gabor_response':gab_r,'optimization_frames':a.opt_frames,'evaluation_frames':a.frames,'iterations':a.iterations,'comparison_gif':str(out)}; out.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
