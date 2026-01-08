import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

import wandb
import os
import argparse
from tqdm import tqdm

from utils.dataset_helper import create_dataloaders, create_train_val_dataloaders
from vae_model import GaussianVAE, vae_loss_sinkhorn, gaussian_visual_loss
from utils.training_utils import get_warmup_cosine_scheduler
from utils.vae_utils import sample_from_latent, save_target_visualization, visualize_reconstruction
import yaml

def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg, kl_weight=0.001, sinkhorn_eps=0.1, recon_weight=0.1, l1_weight=0.1, ssim_weight=0.1):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_l1_loss = 0
    epoch_ssim_loss = 0
    valid_batches = 0

    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} | Eps: {sinkhorn_eps:.5f}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        x = batch.to(device)  # [B, N, 8]
        
        # Determine if we should sync gradients (only on the step where optimizer.step() runs)
        # We sync on the LAST sub-step of the accumulation.
        # If grad_accumulation is 1, we always sync.
        should_sync = ((batch_idx + 1) % cfg["grad_accumulation"] == 0)
        
        # Use no_sync() context if we are NOT syncing (and if model is DDP)
        # If the model is not DDP, no_sync is not available, but nullcontext handles it if we structured perfectly.
        # However, DDP wrapper adds no_sync. If not distributed, model is raw.
        # We need to be careful.
        
        my_context = model.no_sync() if (isinstance(model, DDP) and not should_sync) else nullcontext()

        with my_context:
            # Forward pass - DETERMINISTIC for overfitting
            x_recon, mu, logvar = model(x, use_sampling=True)

            # Compute VAE loss using Sinkhorn with annealing
            vae_loss, recon_loss, kl_loss = vae_loss_sinkhorn(
                x_recon, x, mu, logvar, 
                recon_weight=recon_weight, # Reduced reconstruction weight to balance with LPIPS
                kl_weight=kl_weight,
                sinkhorn_epsilon=sinkhorn_eps
            )

            # Compute visual losses
            # l1_loss_val, ssim_loss_val = gaussian_visual_loss(x, x_recon, device)
            l1_loss_val, ssim_loss_val = torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device)   

            total_loss = vae_loss + l1_weight * l1_loss_val + ssim_weight * ssim_loss_val
            
            # Backward pass with gradient accumulation
            loss_scaled = total_loss / cfg["grad_accumulation"]
            loss_scaled.backward()
        
        # Gradient accumulation step
        if should_sync:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_norm"])
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate losses
        # Log metrics using maximum weights to avoid artifacts from annealing
        max_kl_weight = cfg["train"].get("max_kl_weight", 1.0)
        max_lpips_weight = cfg["train"].get("lpips_weight", 0.1)
        
        epoch_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item() * recon_weight
        epoch_kl_loss += kl_loss.item() * max_kl_weight
        epoch_l1_loss += l1_loss_val.item()
        epoch_ssim_loss += ssim_loss_val.item()
        valid_batches += 1
        
        # Update progress bar
        # Display weighted versions to see their impact on total loss
        weighted_visual = l1_weight * l1_loss_val.item() + ssim_weight * ssim_loss_val.item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item() * recon_weight:.4f}',
            'kl': f'{kl_loss.item() * max_kl_weight:.6f}',
            'vis': f'{weighted_visual:.4f}',
            'eps': f'{sinkhorn_eps:.5f}'
        })
    
    # Final gradient step if there are remaining accumulated gradients
    if valid_batches % cfg["grad_accumulation"] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_norm"])
        optimizer.step()
        optimizer.zero_grad()
    
    # Return average losses
    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_recon = epoch_recon_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_kl = epoch_kl_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_l1 = epoch_l1_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_ssim = epoch_ssim_loss / valid_batches if valid_batches > 0 else float('inf')
    
    return avg_loss, avg_recon, avg_kl, avg_l1, avg_ssim

def validate_one_epoch(model, dataloader, device, cfg, sinkhorn_eps=0.1, recon_weight=0.1, l1_weight=0.1, ssim_weight=0.1, kl_weight=0.001):
    model.eval()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_l1_loss = 0
    epoch_ssim_loss = 0
    valid_batches = 0
    
    # Added for consistency with train_one_epoch logging
    max_kl_weight = cfg["train"].get("max_kl_weight", 1.0)

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            x_recon, mu, logvar = model(x, use_sampling=True)
            
            vae_loss, recon_loss, kl_loss = vae_loss_sinkhorn(
                x_recon, x, mu, logvar,
                recon_weight=recon_weight,
                kl_weight=kl_weight,
                sinkhorn_epsilon=sinkhorn_eps
            )
            
            l1_loss_val, ssim_loss_val = gaussian_visual_loss(x, x_recon, device)
            total_loss = vae_loss + l1_weight * l1_loss_val + ssim_weight * ssim_loss_val
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item() * recon_weight
            epoch_kl_loss += kl_loss.item() * max_kl_weight
            epoch_l1_loss += l1_loss_val.item()
            epoch_ssim_loss += ssim_loss_val.item()
            valid_batches += 1
            
    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_recon = epoch_recon_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_kl = epoch_kl_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_l1 = epoch_l1_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_ssim = epoch_ssim_loss / valid_batches if valid_batches > 0 else float('inf')
    
    return avg_loss, avg_recon, avg_kl, avg_l1, avg_ssim


def train_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_training.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    overfit = cfg.get("overfit", False)

    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (dist.get_rank() == 0)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if is_main_process:
        print(f"Starting training on {device}...")
    
    train_loader, val_loader, train_sampler, val_sampler = create_train_val_dataloaders(
        cfg["data_dir"], 
        batch_size=cfg["batch_size"], 
        validation_split=cfg.get("validation_split", 0.05), # Default 5%
        shuffle=True, 
        augment=cfg.get("augment", False),
        is_distributed=is_distributed
    )
    
    if is_main_process:
        save_target_visualization(val_loader, device, cfg)

    # 2. Model
    model = GaussianVAE(
        num_gaussians=cfg["model"]["num_gaussians"],
        input_dim=cfg["model"]["input_dim"],
        latent_dim=cfg["model"]["model_dim"],
    ).to(device)

    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params / 1e6:.2f}M Params")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["base_lr"], weight_decay=1e-4)
    
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=cfg["train"]["lr_warmup_epochs"], 
        max_epochs=cfg["train"]["max_epochs"]
    )

    if is_main_process:
        wandb.init(project="gaussian-vae", config=cfg)
    best_val_loss = float('inf')

    wandb.watch(model, log="all", log_freq=10)

    #Save the recons of the initial untrained model
    if is_main_process:
        raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore
        visualize_reconstruction(raw_model, val_loader, device, cfg, epoch=0)

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        target_kl = cfg["train"].get("max_kl_weight", 1.0)
        kl_warmup = cfg["train"].get("kl_warmup_epochs", 500)
        kl_weight = min(target_kl, target_kl * epoch / kl_warmup)

        sinkhorn_eps = max(0.001, 0.5 * ((1 - 1 / cfg["train"].get("sinkhorn_warmup_epochs", 200)) ** epoch)) 
        
        recon_weight = cfg["train"].get("recon_weight", 0.1)

        target_l1 = cfg["train"].get("l1_weight", 0.1)
        target_ssim = cfg["train"].get("ssim_weight", 0.1)
        vis_warmup = cfg["train"].get("vis_warmup_epochs", 1000)

        l1_weight = min(target_l1, target_l1 * epoch / vis_warmup)
        ssim_weight = min(target_ssim, target_ssim * epoch / vis_warmup)

        current_lr = optimizer.param_groups[0]['lr']
              
        avg_loss, avg_recon, avg_kl, avg_l1, avg_ssim = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, 
            kl_weight=kl_weight, 
            sinkhorn_eps=sinkhorn_eps,
            recon_weight=recon_weight,
            l1_weight=l1_weight,
            ssim_weight=ssim_weight
        )
        
        val_loss, val_recon, val_kl, val_l1, val_ssim = validate_one_epoch(
            model, val_loader, device, cfg, 
            sinkhorn_eps=0.1, # Use low epsilon for validation to see true performance
            recon_weight=recon_weight,
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            kl_weight=kl_weight
        )
        
        if is_main_process:
            wandb.log({
                "val/total_loss": val_loss,
                "val/recon_loss": val_recon,
                "val/kl_loss": val_kl,
                "val/l1_loss": val_l1,
                "val/ssim_loss": val_ssim,
                "train/total_loss": avg_loss,
                "train/recon_loss": avg_recon,
                "train/kl_loss": avg_kl,
                "train/l1_loss": avg_l1,
                "train/ssim_loss": avg_ssim,
                "train/lr": current_lr,
                "train/kl_weight": kl_weight,
                "train/l1_weight": l1_weight,
                "train/ssim_weight": ssim_weight,
                "train/sinkhorn_eps": sinkhorn_eps,
                "epoch": epoch,
            })
            
            # Save checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{cfg['output_dir']}/best_gaussian_vae.pth")

            # Save periodic checkpoint
            if epoch % cfg["logging"].get("checkpoint_save_rate", 500) == 0:
                 torch.save(model.state_dict(), f"{cfg['output_dir']}/checkpoint_epoch_{epoch}.pth")
            
            print(f"Epoch {epoch}/{cfg['train']['max_epochs']} | "
                  f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, L1: {avg_l1:.4f}, SSIM: {avg_ssim:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.6f}, L1: {val_l1:.4f}, SSIM: {val_ssim:.4f}) | "
                  f"LR: {current_lr:.6f}")

            # Safe model access (handle both DDP and regular)
            raw_model = model.module if hasattr(model, "module") else model # type: ignore

            if epoch % cfg["logging"]["sample_save_rate"] == 0:
                sample_from_latent(raw_model, device, cfg, num_samples=5, epoch=epoch)

            if epoch % cfg["logging"].get("reconstruct_save_rate", 100) == 0:
                visualize_reconstruction(raw_model, val_loader, device, cfg, epoch=epoch)


        lr_scheduler.step()

    if is_main_process:
        raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore
        torch.save(raw_model.state_dict(), "final_vae_model.pth")
        print("Training Complete. Final model saved.")
        wandb.finish()

if __name__ == "__main__":
    train_vae()
