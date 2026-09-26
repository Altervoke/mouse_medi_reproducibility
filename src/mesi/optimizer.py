"""Static latent MESI optimizer."""
import random
import numpy as np
import torch
import torch.nn.functional as F

def generate_mesi_batch(vae, model, readout_indices, iterations=10, lr=10.0,
                        seed=42, device="cuda", frames=20):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    vae.enable_gradient_checkpointing()
    batch=len(readout_indices)
    if batch == 0:
        raise ValueError("readout_indices must not be empty")
    latents=(torch.randn(batch,4,1,36,64,device=device)*0.02+0.5).requires_grad_(True)
    opt=torch.optim.SGD([latents],lr=lr,momentum=0.9)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=iterations,eta_min=lr*0.1)
    targets=torch.as_tensor(readout_indices,dtype=torch.long,device=device)
    for _ in range(iterations):
        opt.zero_grad(); noisy=latents+torch.randn_like(latents)*0.05
        decoded=[]
        for i in range(batch):
            d=torch.utils.checkpoint.checkpoint(vae.decode,torch.clamp(noisy[i,:,0]/1.0,-10,10).float().unsqueeze(0),use_reentrant=False).sample
            d=F.interpolate(d,(144,256),mode="bilinear",align_corners=False)
            g=0.299*d[:,0:1]+0.587*d[:,1:2]+0.114*d[:,2:3]
            decoded.append(torch.clamp((torch.clamp(g,-1,1)+1)*0.5,0,1))
        image=torch.cat(decoded,0); model.reset(); x=image.unsqueeze(0).repeat(frames,1,1,1,1)
        p=torch.zeros(batch,2,device=device); m=torch.zeros(batch,2,device=device)
        response=torch.stack([model(x[t],p,m) for t in range(frames)])
        # Select the target readout for each batch item without advanced
        # indexing ambiguities (the production path normally uses batch=1).
        target=response[:, torch.arange(batch, device=device), targets].mean(0)
        smooth=(latents[:,:,:,1:]-latents[:,:,:,:-1]).square().mean()+ (latents[:,:,:,:,1:]-latents[:,:,:,:,:-1]).square().mean()
        loss=(-torch.log(torch.relu(target)+1e-4)*3).sum()+500*smooth
        loss.backward(); torch.nn.utils.clip_grad_norm_([latents],0.1); opt.step(); latents.data.clamp_(-5,5); sched.step()
    with torch.no_grad():
        d=torch.cat([vae.decode(torch.clamp(latents[i,:,0]/1.0,-10,10).float().unsqueeze(0)).sample for i in range(batch)], dim=0)
        d=F.interpolate(d,(144,256),mode="bilinear",align_corners=False); g=0.299*d[:,0]+0.587*d[:,1]+0.114*d[:,2]
        return (torch.clamp((torch.clamp(g,-1,1)+1)*127.5,0,255).byte().cpu().numpy())
