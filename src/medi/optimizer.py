import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

VAE_SCALING_FACTOR = 1.0
LATENT_CHANNELS = 4
LATENT_H = 36
LATENT_W = 64

def generate_medi_batch(
    vae,
    model,
    readout_indices,
    total_frames=60,
    chunk_size=15,
    chunk_overlap=8,
    iterations=20,
    lr=10.0,
    seed=42,
    device='cuda',
    response_reduction='mean',
):
    if total_frames <= 0:
        raise ValueError('total_frames must be positive')
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError('chunk_overlap must satisfy 0 <= overlap < chunk_size')
    readout_indices = torch.as_tensor(
        readout_indices, dtype=torch.long, device=device
    ).reshape(-1)
    if not len(readout_indices):
        raise ValueError('readout_indices must not be empty')
    if response_reduction not in {'mean', 'max'}:
        raise ValueError("response_reduction must be 'mean' or 'max'")
    batch_size = len(readout_indices)
    vae.enable_gradient_checkpointing()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < total_frames:
        # Keep the final optimization chunk full-sized, matching the original
        # implementation; the stitched sequence is cropped to total_frames
        # only after all chunks have been optimized.
        end = start + chunk_size
        chunks.append((start, end))
        if end >= total_frames:
            break
        start += step
    
    prev_chunk_latents = None 
    final_stitched_latents_list = []
    latent_init = (
        torch.randn(
            (batch_size, LATENT_CHANNELS, 1, LATENT_H, LATENT_W),
            device=device,
        )
        * 0.02
        + 0.5
    )

    for i, (c_start, c_end) in enumerate(chunks):
        current_chunk_len = c_end - c_start
        model.reset()
        
        if i == 0:
            latents_init = latent_init.repeat(1, 1, current_chunk_len, 1, 1).clone().detach()

        else:
            overlap_len = min(chunk_overlap, prev_chunk_latents.shape[2])
            overlap_section = prev_chunk_latents[:, :, -overlap_len:, :, :].clone().detach()
            
            new_len = current_chunk_len - overlap_len
            if new_len > 0:
                new_section = latent_init.repeat(1, 1, new_len, 1, 1).clone().detach()
                latents_init = torch.cat([overlap_section, new_section], dim=2)
            else:
                latents_init = overlap_section

        latents = latents_init.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.SGD([latents], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr*0.1)
        
        pbar = tqdm(range(iterations), desc=f"Chunk {i}", leave=False)
        for it in pbar:
            optimizer.zero_grad()
            
            # 1. Latent Robustness
            noise = torch.randn_like(latents) * 0.05
            noisy_latents = latents + noise
            
            # 2. Decode
            dec_list = []
            vae.float()
            for frame_idx in range(noisy_latents.shape[2]):
                frame_latents = torch.clamp(
                    noisy_latents[:, :, frame_idx] / VAE_SCALING_FACTOR,
                    -10,
                    10,
                ).float()
                d = torch.utils.checkpoint.checkpoint(
                    vae.decode, frame_latents, use_reentrant=False
                ).sample
                dec_list.append(d)
            decoded = torch.stack(dec_list, dim=0)
            
            # 3. Preprocess
            frame_count, _, channels, decoded_h, decoded_w = decoded.shape
            dec_sz = F.interpolate(
                decoded.reshape(frame_count * batch_size, channels, decoded_h, decoded_w),
                (144, 256),
                mode='bilinear',
                align_corners=False,
            ).reshape(frame_count, batch_size, channels, 144, 256)
            dec_gray = (
                0.299 * dec_sz[:, :, 0:1]
                + 0.587 * dec_sz[:, :, 1:2]
                + 0.114 * dec_sz[:, :, 2:3]
            )
            dec_clamped = torch.clamp(dec_gray, -1, 1)
            model_in = torch.clamp((dec_clamped+1.0)*0.5, 0.0, 1.0)
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = model_in.dtype
            model_input_for_model = model_in.to(dtype=model_dtype)
            
            # 4. Forward Pass
            model.reset()
            
            N = model_in.shape[0]
            dummy_p = torch.zeros(batch_size, 2, device=device)
            dummy_m = torch.zeros(batch_size, 2, device=device)
            
            resps = []
            for t in range(N):
                resps.append(model(model_input_for_model[t], dummy_p, dummy_m))
            resp = torch.stack(resps, dim=0)
            
            # 5. Losses
            target_resp = _select_batched_target_responses(
                resp, readout_indices, reduction=response_reduction
            )
            
            safe_resp = torch.relu(target_resp) + 1e-4
            loss_log_resp = (-torch.log(safe_resp) * 3.0).sum()
            
            d_latents = latents[:, :, 1:] - latents[:, :, :-1]
            loss_temp_latent = (
                d_latents.square().flatten(1).mean(dim=1) * 2000.0
            ).sum()
            
            d_pixels = model_in[1:] - model_in[:-1]
            loss_temp_pixel = (
                d_pixels.permute(1, 0, 2, 3, 4).flatten(1).square().mean(dim=1)
                * 5000.0
            ).sum()
            
            d_lat_y = latents[:,:,:,1:,:] - latents[:,:,:,:-1,:]
            d_lat_x = latents[:,:,:,:,1:] - latents[:,:,:,:,:-1]
            loss_spatial = (
                d_lat_y.square().flatten(1).mean(dim=1)
                + d_lat_x.square().flatten(1).mean(dim=1)
            ).sum() * 500.0
            
            total_loss = loss_log_resp + loss_temp_latent + loss_temp_pixel + loss_spatial
            total_loss.backward()
            
            # 6. Gradient Masking & Clipping
            if chunk_overlap > 0 and i > 0:
                scales = torch.linspace(0.0, 1.0, steps=chunk_overlap, device=latents.grad.device)
                scales = scales.view(1, 1, chunk_overlap, 1, 1)
                latents.grad[:, :, 0:chunk_overlap, :, :] *= scales
                    
            latents.grad[:,:,0:1,:,:] = 0.0
            grad_norms = latents.grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            grad_scales = (0.1 / grad_norms).clamp(max=1.0)
            latents.grad.mul_(grad_scales.view(batch_size, 1, 1, 1, 1))
            optimizer.step()
            scheduler.step()
            latents.data.clamp_(-5.0, 5.0)
        
        prev_chunk_latents = latents.detach().clone()
        len_to_keep = current_chunk_len
        if i < len(chunks) - 1:
            len_to_keep -= chunk_overlap
            
        chunk_latents_final = latents[:, :, :len_to_keep, :, :].detach().cpu()
        final_stitched_latents_list.append(chunk_latents_final)

    final_latents_tensor = torch.cat(final_stitched_latents_list, dim=2)
    
    with torch.no_grad():
        final_latents_tensor = final_latents_tensor.to(device)
        d_list = []
        vae.float()
        for frame_idx in range(final_latents_tensor.shape[2]):
            d_list.append(
                vae.decode(
                    final_latents_tensor[:, :, frame_idx] / VAE_SCALING_FACTOR
                ).sample
            )
        
        vid = torch.stack(d_list, dim=1)
        
        _, frame_count, channels, decoded_h, decoded_w = vid.shape
        vid_144 = F.interpolate(
            vid.reshape(batch_size * frame_count, channels, decoded_h, decoded_w),
            size=(144, 256),
            mode='bilinear',
            align_corners=False,
        ).reshape(batch_size, frame_count, channels, 144, 256)
        vid_gray = (
            0.299 * vid_144[:, :, 0:1]
            + 0.587 * vid_144[:, :, 1:2]
            + 0.114 * vid_144[:, :, 2:3]
        )
        vid_clamped = torch.clamp(vid_gray, -1, 1)
        vid_final = torch.clamp((vid_clamped + 1.0) * 0.5, 0, 1)
        vid_numpy = vid_final.squeeze(2).cpu().numpy()
        vid_uint8 = (vid_numpy * 255).astype(np.uint8)
        
    return vid_uint8[:, :total_frames]


def _select_batched_target_responses(responses, readout_indices, reduction='mean'):
    """Reduce each batch item's temporal response for its target readout."""
    if responses.ndim != 3:
        raise ValueError('responses must have shape [time, batch, readout]')
    readout_indices = torch.as_tensor(
        readout_indices, dtype=torch.long, device=responses.device
    ).reshape(-1)
    if responses.shape[1] != len(readout_indices):
        raise ValueError('one readout index is required for each batch item')
    if reduction not in {'mean', 'max'}:
        raise ValueError("reduction must be 'mean' or 'max'")
    batch_indices = torch.arange(responses.shape[1], device=responses.device)
    trajectories = responses[:, batch_indices, readout_indices]
    if reduction == 'mean':
        return trajectories.mean(dim=0)
    return trajectories.max(dim=0).values
