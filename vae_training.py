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
from vae_model import GaussianVAE, vae_loss_sinkhorn, gaussian_lpips_loss
from utils.training_utils import get_warmup_cosine_scheduler
from utils.vae_utils import sample_from_latent, save_target_visualization, visualize_reconstruction
import yaml

def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg, kl_weight=0.001, lpips_weight=0.1, sinkhorn_eps=0.1, recon_weight=0.1):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_lpips_loss = 0
    valid_batches = 0
    
    overfit = cfg.get("overfit", False)

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
            x_recon, mu, logvar = model(x, use_sampling=True if not overfit else False)

            # Compute VAE loss using Sinkhorn with annealing
            vae_loss, recon_loss, kl_loss = vae_loss_sinkhorn(
                x_recon, x, mu, logvar, 
                recon_weight=recon_weight, # Reduced reconstruction weight to balance with LPIPS
                kl_weight=kl_weight,
                sinkhorn_epsilon=sinkhorn_eps
            )
            
            lpips_val = gaussian_lpips_loss(x, x_recon, device)
            
            total_loss = vae_loss + lpips_weight * lpips_val
            
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
        
        epoch_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item() * recon_weight
        epoch_kl_loss += kl_loss.item() * max_kl_weight
        epoch_lpips_loss += lpips_val.item() * lpips_weight
        valid_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item() * recon_weight:.4f}',
            'kl': f'{kl_loss.item() * max_kl_weight:.6f}',
            'lpips': f'{lpips_val.item() * lpips_weight:.4f}',
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
    avg_lpips = epoch_lpips_loss / valid_batches if valid_batches > 0 else float('inf')
    
    return avg_loss, avg_recon, avg_kl, avg_lpips

def validate_one_epoch(model, dataloader, device, cfg, sinkhorn_eps=0.1, recon_weight=0.1, lpips_weight=0.1):
    model.eval()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_lpips_loss = 0
    valid_batches = 0
    
    lpips_weight = cfg.get("lpips_weight", lpips_weight) # Fallback to argument or config override if present in root (legacy)

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            x_recon, mu, logvar = model(x, use_sampling=True)
            
            vae_loss, recon_loss, kl_loss = vae_loss_sinkhorn(
                x_recon, x, mu, logvar,
                recon_weight=recon_weight,
                kl_weight=cfg.get("validation_kl_weight", 0.001), # validation weight could be fixed
                sinkhorn_epsilon=sinkhorn_eps
            )
            
            lpips_val = gaussian_lpips_loss(x, x_recon, device)
            total_loss = vae_loss + lpips_weight * lpips_val
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item() * recon_weight
            epoch_kl_loss += kl_loss.item() * cfg.get("validation_kl_weight", 0.001)
            epoch_lpips_loss += lpips_val.item() * lpips_weight
            valid_batches += 1
            
    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_recon = epoch_recon_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_kl = epoch_kl_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_lpips = epoch_lpips_loss / valid_batches if valid_batches > 0 else float('inf')
    
    model.train()
    return avg_loss, avg_recon, avg_kl, avg_lpips


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
        shuffle=True if not overfit else False, 
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

    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["base_lr"], weight_decay=1e-4 if not overfit else 0.0)
    
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=cfg["train"]["warmup_epochs"], 
        max_epochs=cfg["train"]["max_epochs"]
    )

    if is_main_process:
        wandb.init(project="gaussian-vae", config=cfg)
    best_val_loss = float('inf')

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        target_kl = cfg["train"].get("max_kl_weight", 1.0)
        warmup_epochs = cfg["train"].get("warmup_epochs", 200) # Use the same warmup
        # Or should we have a separate kl_warmup_epochs? existing code used warmup_epochs hardcoded/from variable context
        # In existing code: warmup_epochs = 200. "train" block has warmup_epochs: 150.
        # The code had `warmup_epochs = 200` locally defined.
        
        # Let's clean this up.
        # Existing code:
        # target_kl = 1.0
        # warmup_epochs = 200
        # kl_weight = min(target_kl, target_kl * epoch / warmup_epochs) if not overfit else 0.0
        
        kl_warmup = 200 # keeping hardcoded default as fallback or separate config? keeping as before but using target_kl
        
        kl_weight = min(target_kl, target_kl * epoch / kl_warmup) if not overfit else 0.0
        sinkhorn_eps = max(0.001, 0.5 * (0.995 ** epoch)) 
        
        recon_weight = cfg["train"].get("recon_weight", 0.1)
        lpips_weight = cfg["train"].get("lpips_weight", 0.1)
              
        avg_loss, avg_recon, avg_kl, avg_lpips = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, 
            kl_weight=kl_weight, 
            lpips_weight=lpips_weight,
            sinkhorn_eps=sinkhorn_eps,
            recon_weight=recon_weight
        )
        current_lr = optimizer.param_groups[0]['lr']
        
        val_loss, val_recon, val_kl, val_lpips = validate_one_epoch(
            model, val_loader, device, cfg, sinkhorn_eps=sinkhorn_eps,
            recon_weight=recon_weight,
            lpips_weight=lpips_weight
        )
    
        if is_main_process:
            raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if epoch > 50:
                    torch.save(raw_model.state_dict(), "best_gaussian_vae.pth")
                    print(f"--> New best model saved (Val Loss: {best_val_loss:.4f})")
            
            # Log
            wandb.log({
                "epoch": epoch, 
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "kl_loss": avg_kl,
                "lpips_loss": avg_lpips,
                "val_loss": val_loss,
                "val_recon_loss": val_recon,
                "val_kl_loss": val_kl,
                "val_lpips_loss": val_lpips,
                "kl_weight": kl_weight,
                "sinkhorn_eps": sinkhorn_eps,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss
            })
            print(f"Epoch {epoch}/{cfg['train']['max_epochs']} | "
                  f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, LPIPS: {avg_lpips:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.6f}, LPIPS: {val_lpips:.4f}) | "
                  f"LR: {current_lr:.6f}")

            if epoch % cfg["logging"]["sample_save_rate"] == 0:
                sample_from_latent(raw_model, device, cfg, num_samples=5, epoch=epoch)

            if epoch % cfg["logging"].get("reconstruct_save_rate", 100) == 0:
                visualize_reconstruction(raw_model, val_loader, device, cfg, epoch=epoch)

            if epoch % 500 == 0:
                torch.save(raw_model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

        lr_scheduler.step()

    if is_main_process:
        raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore
        torch.save(raw_model.state_dict(), "final_vae_model.pth")
        print("Training Complete. Final model saved.")
        wandb.finish()

if __name__ == "__main__":
    train_vae()
