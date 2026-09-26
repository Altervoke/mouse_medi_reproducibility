from pathlib import Path
import argparse, json, sys, time, math
import numpy as np
import imageio.v2 as imageio
import torch
from PIL import Image, ImageDraw, ImageFont

def load_model(project_root, session, scan, device):
    project_root = Path(project_root)
    sys.path.insert(0, str(project_root))
    try:
        from generation.generator import load_foundation_model
        model, units = load_foundation_model(session, scan, device=device, dtype='float32', return_units=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, units
    except ImportError:
        # Licensed FNN runtime fallback used by the server pilot.
        import os
        fnn_root = os.environ.get('FNN_ROOT')
        if fnn_root and fnn_root not in sys.path:
            sys.path.insert(0, fnn_root)
        from fnn import microns
        model, units = microns.scan(session=session, scan_idx=scan, cuda=device.startswith('cuda'), directory=str(project_root))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, units

def load_gif(path, frames=60):
    arr = np.stack([np.asarray(x.convert('L') if hasattr(x, 'convert') else x)
                    for x in imageio.mimread(path)[:frames]])
    if arr.ndim == 4: arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0

def response(model, video, readout_index, device):
    x = torch.from_numpy(video[:, None]).to(device=device, dtype=torch.float32)
    p = torch.zeros((len(video), 2), device=device); m = torch.zeros((len(video), 2), device=device)
    model.reset()
    y = torch.cat([model(x[t:t+1], p[t:t+1], m[t:t+1]) for t in range(len(video))], 0)
    return float(y[10:, readout_index].mean().detach().cpu())

class FitGaborDynamic(torch.nn.Module):
    """fitgabor spatial Gabor plus conservative smooth temporal terms."""
    def __init__(self, h=144, w=256, frames=30, device='cuda',
                 temporal_frequency_bound=.015, velocity_bound=.003):
        super().__init__(); self.frames = frames
        self.temporal_frequency_bound = float(temporal_frequency_bound)
        self.velocity_bound = float(velocity_bound)
        yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing='ij')
        self.register_buffer('xx', xx); self.register_buffer('yy', yy)
        # fitgabor initialization/ranges.
        self.theta = torch.nn.Parameter(torch.rand(1,device=device)*4*math.pi-2*math.pi)
        self.sigma = torch.nn.Parameter(torch.rand(1,device=device)*.05+.15)
        self.Lambda = torch.nn.Parameter(torch.rand(1,device=device)*.2+.5)
        self.psi = torch.nn.Parameter(torch.rand(1,device=device)*math.pi/2)
        self.gamma = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.contrast = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.center = torch.nn.Parameter(torch.zeros(2,device=device))
        # Conservative dynamics: >=~33-frame temporal period and <=0.3 image-width drift.
        self.temporal_frequency = torch.nn.Parameter(torch.zeros(1,device=device))
        self.velocity = torch.nn.Parameter(torch.zeros(2,device=device))
    def project_(self):
        with torch.no_grad():
            self.theta.clamp_(-2*math.pi,2*math.pi); self.sigma.clamp_(.13,.25)
            self.Lambda.clamp_(.2,2.0); self.psi.clamp_(-math.pi,math.pi)
            self.gamma.clamp_(0.5, 2.0)
            self.contrast.clamp_(0.1, 2.0)
            self.center.clamp_(-.55,.55)
            self.temporal_frequency.clamp_(-self.temporal_frequency_bound,
                                           self.temporal_frequency_bound)
            self.velocity.clamp_(-self.velocity_bound, self.velocity_bound)
    def forward(self):
        c,s=torch.cos(self.theta),torch.sin(self.theta)
        t=torch.arange(self.frames,device=self.xx.device,dtype=self.xx.dtype)[:,None,None]
        cx=self.center[0]+self.velocity[0]*t; cy=self.center[1]+self.velocity[1]*t
        dx=self.xx[None]-cx; dy=self.yy[None]-cy
        xp=dx*c+dy*s; yp=-dx*s+dy*c
        env=torch.exp(-.5*(xp**2/self.sigma**2+yp**2/(self.sigma*self.gamma).clamp_min(1e-4)**2))
        phase=2*math.pi*xp/self.Lambda+self.psi+2*math.pi*self.temporal_frequency*t
        raw=env*torch.cos(phase)
        raw=self.contrast*.1*raw/(raw.std()+1e-8)
        return (raw.clamp(-1,1)+1)/2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,required=True); ap.add_argument('--session',type=int,required=True); ap.add_argument('--scan',type=int,required=True); ap.add_argument('--readout',type=int,required=True); ap.add_argument('--medi',required=True); ap.add_argument('--out',required=True); ap.add_argument('--iterations',type=int,default=30); ap.add_argument('--frames',type=int,default=60); ap.add_argument('--opt-frames',type=int,default=60); ap.add_argument('--device',default='cuda'); ap.add_argument('--log-every',type=int,default=1); ap.add_argument('--temporal-frequency-bound',type=float,default=.32); ap.add_argument('--velocity-bound',type=float,default=.006); a=ap.parse_args()
    if a.opt_frames != a.frames:
        raise ValueError('production dynamic-Gabor optimization and evaluation must use the same frame count')
    device=a.device if torch.cuda.is_available() or a.device=='cpu' else 'cpu'; model,units=load_model(a.project_root,a.session,a.scan,device)
    if hasattr(units,'columns') and 'readout_id' in units.columns: ids=[int(x) for x in units['readout_id'].tolist()]
    elif getattr(units.index,'name',None)=='readout_id': ids=[int(x) for x in units.index.tolist()]
    else: raise ValueError('Cannot map readout_id: expected readout_id column or index')
    if a.readout not in ids: raise ValueError(f'readout_id {a.readout} is not present in scan {a.session}/{a.scan}')
    idx=ids.index(a.readout)
    medi=load_gif(a.medi,a.frames); medi_resp=response(model,medi,idx,device); torch.manual_seed(7000+a.readout); g=FitGaborDynamic(frames=a.opt_frames,device=device,temporal_frequency_bound=a.temporal_frequency_bound,velocity_bound=a.velocity_bound); opt=torch.optim.Adam(g.parameters(),lr=.05); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5); start=time.time(); best=(-float('inf'),None)
    for i in range(a.iterations):
        opt.zero_grad(); vid=g(); model.reset(); x=vid[:,None]; p=torch.zeros((len(vid),2),device=device); m=torch.zeros((len(vid),2),device=device); y=torch.cat([model(x[t:t+1],p[t:t+1],m[t:t+1]) for t in range(len(vid))],0); loss=-y[10:,idx].mean(); score=-float(loss.detach().cpu())
        # The logged score belongs to the parameters used to generate `vid`.
        # Save that exact state before applying the optimizer update.
        if score>best[0]: best=(score,{k:v.detach().clone() for k,v in g.named_parameters()})
        loss.backward(); opt.step(); g.project_(); sched.step(score)
        if (i+1)%a.log_every==0:
            if device.startswith('cuda'): torch.cuda.synchronize()
            elapsed=time.time()-start; print(f'[fitgabor-dynamic] iter {i+1}/{a.iterations} response={score:.4f} elapsed={elapsed:.1f}s ETA={elapsed/(i+1)*(a.iterations-i-1):.1f}s',flush=True)
    with torch.no_grad():
        for k,v in best[1].items(): getattr(g,k).copy_(v)
        g.frames=a.frames; gab=g().cpu().numpy()
    gab_resp=response(model,gab,idx,device)
    # One GIF: left MEDI, right Gabor, synchronized frames.
    panels=[]
    try: font=ImageFont.truetype('DejaVuSans-Bold.ttf',18)
    except Exception: font=ImageFont.load_default()
    for t in range(a.frames):
        left=(medi[t]*255).astype(np.uint8); right=(gab[t]*255).astype(np.uint8); sep=np.full((left.shape[0],8),240,np.uint8); canvas=np.concatenate([left,sep,right],axis=1)
        labelled=np.full((canvas.shape[0]+30,canvas.shape[1]),245,np.uint8); labelled[30:]=canvas
        im=Image.fromarray(labelled); draw=ImageDraw.Draw(im); draw.text((left.shape[1]//2-25,6),'MEDI',fill=20,font=font); draw.text((left.shape[1]+8+right.shape[1]//2-65,6),'fitgabor dynamic',fill=20,font=font)
        panels.append(np.asarray(im))
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); imageio.mimsave(out,panels,fps=30,loop=0)
    parameters={}
    for k,v in g.named_parameters():
        values=v.detach().cpu().reshape(-1).tolist()
        parameters[k]=values[0] if len(values)==1 else values
    meta={'session':a.session,'scan':a.scan,'readout':a.readout,'medi_path':a.medi,'comparison_gif':str(out),'optimization_frames':a.opt_frames,'evaluation_frames':a.frames,'iterations':a.iterations,'temporal_frequency_bound':a.temporal_frequency_bound,'velocity_bound':a.velocity_bound,'medi_response_last50':medi_resp,'dynamic_gabor_response_last50':gab_resp,'parameters':parameters}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()

