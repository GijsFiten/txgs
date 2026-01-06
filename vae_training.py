import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import os
import argparse
from tqdm import tqdm

from utils.dataset_helper import create_dataloaders, create_train_val_dataloaders
from vae_model import GaussianVAE, vae_loss_sinkhorn
from utils.training_utils import get_warmup_cosine_scheduler
from utils.vae_utils import sample_from_latent, save_target_visualization, visualize_reconstruction
import yaml

def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg, kl_weight=0.001, sinkhorn_eps=0.1):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    valid_batches = 0
    
    overfit = cfg.get("overfit", False)

    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} | Eps: {sinkhorn_eps:.5f}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        x = batch.to(device)  # [B, N, 8]
        
        # Forward pass - DETERMINISTIC for overfitting
        x_recon, mu, logvar = model(x, use_sampling=True if not overfit else False)

        # Compute VAE loss using Sinkhorn with annealing
        loss, recon_loss, kl_loss = vae_loss_sinkhorn(
            x_recon, x, mu, logvar, 
            recon_weight=1.0, 
            kl_weight=kl_weight,
            sinkhorn_epsilon=sinkhorn_eps
        )
        
        # Backward pass with gradient accumulation
        loss = loss / cfg["grad_accumulation"]
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % cfg["grad_accumulation"] == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_norm"])
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate losses
        epoch_loss += loss.item() * cfg["grad_accumulation"]
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        valid_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * cfg["grad_accumulation"]:.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.6f}',
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
    
    return avg_loss, avg_recon, avg_kl

def validate_one_epoch(model, dataloader, device, cfg, sinkhorn_eps=0.1):
    model.eval()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    valid_batches = 0
    
    # Don't use overfit logic during validation, always sample normally
    # But usually for validation we might want deterministic behavior?
    # standard VAE validation uses sampling to estimate NLL, 
    # but for reconstruction visualization we use mean.
    # Here we'll stick to sampling for loss calculation consistency.
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(x, use_sampling=True)

            # Compute VAE loss 
            # Note: We use fixed weights for validation logging usually, or the same as training?
            # It's better to report the unweighted components separateley.
            # We'll use the same sinkhorn_eps as current training epoch for fair comparison
            
            loss, recon_loss, kl_loss = vae_loss_sinkhorn(
                x_recon, x, mu, logvar,
                recon_weight=1.0,
                kl_weight=cfg.get("validation_kl_weight", 0.001), # validation weight could be fixed
                sinkhorn_epsilon=sinkhorn_eps
            )
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            valid_batches += 1
            
    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_recon = epoch_recon_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_kl = epoch_kl_loss / valid_batches if valid_batches > 0 else float('inf')
    
    model.train()
    return avg_loss, avg_recon, avg_kl


def main():
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

    # 1. Data
    # dataloader, sampler = create_dataloaders(
    #     cfg["data_dir"], 
    #     batch_size=cfg["batch_size"], 
    #     shuffle=True if not overfit else False, 
    #     augment=False,
    #     is_distributed=is_distributed # Pass the flag
    # )
    
    train_loader, val_loader, train_sampler, val_sampler = create_train_val_dataloaders(
        cfg["data_dir"], 
        batch_size=cfg["batch_size"], 
        validation_split=cfg.get("validation_split", 0.05), # Default 5%
        shuffle=True if not overfit else False, 
        augment=cfg.get("augment", False), # Enable augmentation if config says so, default False based on previous logic 
        is_distributed=is_distributed
    )
    
    # Save target visualization (using validation set for stable reference)
    if is_main_process:
        save_target_visualization(val_loader, device, cfg)

    # 2. Model
    model = GaussianVAE(
        num_gaussians=cfg["model"]["num_gaussians"],
        input_dim=cfg["model"]["input_dim"],
        latent_dim=cfg["model"]["model_dim"],
    ).to(device)

    if is_distributed:
        # SyncBatchNorm is crucial if batch_size per GPU is small
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params / 1e6:.2f}M Params")

    # 3. Optimization Components
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["base_lr"], weight_decay=1e-4 if not overfit else 0.0)  # No weight decay for overfitting
    
    # Custom Warmup Scheduler
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=cfg["train"]["warmup_epochs"], 
        max_epochs=cfg["train"]["max_epochs"]
    )

    # 4. Logging
    if is_main_process:
        wandb.init(project="gaussian-vae", config=cfg)
    best_val_loss = float('inf')

    # --- Main Loop ---
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # KL Annealing: gradually increase from 0 to target over first 500 epochs
        kl_weight = min(0.01, 0.01 * epoch / 1000.0) if not overfit else 0.0
        sinkhorn_eps = max(0.001, 0.5 * (0.995 ** epoch)) 
              
        avg_loss, avg_recon, avg_kl = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, kl_weight=kl_weight, sinkhorn_eps=sinkhorn_eps
        )
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation Loop
        val_loss, val_recon, val_kl = validate_one_epoch(
            model, val_loader, device, cfg, sinkhorn_eps=sinkhorn_eps
        )
    
        if is_main_process:
            # Unwrap DDP to get the real model weights and methods
            raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if epoch > 50:
                    # Save raw_model, not model
                    torch.save(raw_model.state_dict(), "best_gaussian_vae.pth")
                    print(f"--> New best model saved (Val Loss: {best_val_loss:.4f})")
            
            # Log
            wandb.log({
                "epoch": epoch, 
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "kl_loss": avg_kl,
                "val_loss": val_loss,
                "val_recon_loss": val_recon,
                "val_kl_loss": val_kl,
                "kl_weight": kl_weight,
                "sinkhorn_eps": sinkhorn_eps,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss
            })
            print(f"Epoch {epoch}/{cfg['train']['max_epochs']} | "
                  f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}) | "
                  f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.6f}) | "
                  f"LR: {current_lr:.6f}")

            # Sampling
            if epoch % cfg["logging"]["sample_save_rate"] == 0:
                # Pass raw_model so we can access .decode()
                sample_from_latent(raw_model, device, cfg, num_samples=5, epoch=epoch)

            # Reconstruction
            if epoch % cfg["logging"].get("reconstruct_save_rate", 100) == 0:
                visualize_reconstruction(raw_model, val_loader, device, cfg, epoch=epoch)

            # Checkpointing every 500 epochs
            if epoch % 500 == 0:
                torch.save(raw_model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    # Final Wrap up
    if is_main_process:
        # Unwrap one last time to be sure
        raw_model: GaussianVAE = model.module if is_distributed else model # type: ignore
        torch.save(raw_model.state_dict(), "final_vae_model.pth")
        print("Training Complete. Final model saved.")
        wandb.finish()

if __name__ == "__main__":
    main()
