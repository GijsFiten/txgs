import os
import torch
from utils.diffusion_data_helper import denormalize_data
from utils.image_utils import render_and_save

def sample_from_latent(model, device, cfg, num_samples=5, epoch=None):
    """Sample from the VAE latent space and render"""
    model.eval()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    with torch.no_grad():
        # Sample random latent vectors from N(0,1)
        z = torch.randn(num_samples, cfg["model"]["model_dim"], device=device)
        
        # Decode to gaussians
        x_sampled = model.decode(z)  # [num_samples, 1000, 8]
        
        print(f"\n[Rendering] Saving {num_samples} VAE samples...")
        for i in range(num_samples):
            sample = x_sampled[i]
            
            # Denormalize
            xy, scale, rot, feat = denormalize_data(
                sample[:, 0:2], sample[:, 2:4], sample[:, 4:5], sample[:, 5:8]
            )
            
            xy = xy.contiguous().float()
            scale = scale.contiguous().float()
            rot = rot.contiguous().float()
            feat = feat.contiguous().float()
            
            img_size = (int(480), int(640))
            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
            filename = f"{cfg['output_dir']}/vae_sample_{i}{epoch_suffix}"
            
            render_and_save(xy, scale, rot, feat, filename, img_size)
    
    model.train()

def visualize_reconstruction(model, dataloader, device, cfg, epoch=None):
    """Reconstruct a batch from the dataloader and render"""
    model.eval()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Get one batch
    batch = next(iter(dataloader))
    batch = batch.to(device)
    
    with torch.no_grad():
        # Forward pass (deterministic)
        x_recon, _, _ = model(batch, use_sampling=False)
        
        print(f"\n[Rendering] Saving reconstruction...")
        # Save first item in batch
        sample = x_recon[0]
        
        # Denormalize
        xy, scale, rot, feat = denormalize_data(
            sample[:, 0:2], sample[:, 2:4], sample[:, 4:5], sample[:, 5:8]
        )
        
        xy = xy.contiguous().float()
        scale = scale.contiguous().float()
        rot = rot.contiguous().float()
        feat = feat.contiguous().float()
        
        img_size = (int(480), int(640))
        epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
        filename = f"{cfg['output_dir']}/vae_recon{epoch_suffix}"
        
        render_and_save(xy, scale, rot, feat, filename, img_size)
    
    model.train()

def save_target_visualization(dataloader, device, cfg):
    """Save the target ground truth image"""
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Get one batch
    batch = next(iter(dataloader))
    batch = batch.to(device)
    
    # We want the first item in the batch
    target = batch[0] # [N, 8]
    
    print(f"\n[Rendering] Saving target ground truth...")
    
    # Denormalize
    xy, scale, rot, feat = denormalize_data(
        target[:, 0:2], target[:, 2:4], target[:, 4:5], target[:, 5:8]
    )
    
    xy = xy.contiguous().float()
    scale = scale.contiguous().float()
    rot = rot.contiguous().float()
    feat = feat.contiguous().float()
    
    img_size = (int(480), int(640))
    filename = f"{cfg['output_dir']}/target_ground_truth"
    
    render_and_save(xy, scale, rot, feat, filename, img_size)