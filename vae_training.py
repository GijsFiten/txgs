import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math
import os
from tqdm import tqdm  # Recommended for progress bars

from utils.dataset_helper import create_dataloaders
from utils.diffusion_data_helper import DiffusionScheduler, denormalize_data
from vae_model import GaussianVAE, vae_loss
from utils.image_utils import render_and_save

# --- Configuration ---
CONFIG = {
    "data_dir": "./data/chairs_1k/",
    "output_dir": "./output/",
    "batch_size": 16,
    "grad_accumulation": 3,
    "model": {
        "num_gaussians": 1000,
        "input_dim": 8,
        "model_dim": 512,
    },
    "train": {
        "max_epochs": 3000,
        "base_lr": 1e-3,        # Slightly lower max LR for stability
        "warmup_epochs": 100,   # Warmup to prevent shock
        "clip_norm": 2,
    },
    "diffusion_steps": 1000,
}

SAMPLE_SAVE_RATE = 100

# --- Utils: Learning Rate Schedule ---
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs):
    """
    Creates a scheduler that linearly warms up from 0 to base_lr,
    then uses cosine decay to 0.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_from_latent(model, device, num_samples=5, epoch=None):
    """Sample from the VAE latent space and render"""
    model.eval()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    with torch.no_grad():
        # Sample random latent vectors from N(0,1)
        z = torch.randn(num_samples, CONFIG["model"]["model_dim"], device=device)
        
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
            filename = f"{CONFIG['output_dir']}/vae_sample_{i}{epoch_suffix}"
            
            render_and_save(xy, scale, rot, feat, filename, img_size)
    
    model.train()


# --- Core: Training Loop ---
# --- Core: Training Loop ---
def train_one_epoch(model, dataloader, optimizer, device, epoch, kl_weight=0.001):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    valid_batches = 0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        x = batch.to(device)  # [B, N, 8]
        
        # Forward pass
        x_recon, mu, logvar = model(x)
        
        # Compute VAE loss
        from vae_model import vae_loss
        loss, recon_loss, kl_loss = vae_loss(
            x_recon, x, mu, logvar, 
            recon_weight=1.0, 
            kl_weight=kl_weight
        )
        
        # Backward pass with gradient accumulation
        loss = loss / CONFIG["grad_accumulation"]
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % CONFIG["grad_accumulation"] == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["train"]["clip_norm"])
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate losses
        epoch_loss += loss.item() * CONFIG["grad_accumulation"]
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        valid_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * CONFIG["grad_accumulation"]:.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.6f}'
        })
    
    # Final gradient step if there are remaining accumulated gradients
    if valid_batches % CONFIG["grad_accumulation"] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["train"]["clip_norm"])
        optimizer.step()
        optimizer.zero_grad()
    
    # Return average losses
    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_recon = epoch_recon_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_kl = epoch_kl_loss / valid_batches if valid_batches > 0 else float('inf')
    
    return avg_loss, avg_recon, avg_kl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")

    # 1. Data
    dataloader = create_dataloaders(CONFIG["data_dir"], batch_size=CONFIG["batch_size"])

    # 2. Model
    model = GaussianVAE(
        num_gaussians=CONFIG["model"]["num_gaussians"],
        input_dim=CONFIG["model"]["input_dim"],
        latent_dim=CONFIG["model"]["model_dim"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params / 1e6:.2f}M Params")

    # 3. Optimization Components
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["train"]["base_lr"], weight_decay=1e-4)
    
    # Custom Warmup Scheduler
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=CONFIG["train"]["warmup_epochs"], 
        max_epochs=CONFIG["train"]["max_epochs"]
    )

    # 4. Logging
    wandb.init(project="gaussian-vae", config=CONFIG)
    best_loss = float('inf')

    # --- Main Loop ---
    for epoch in range(1, CONFIG["train"]["max_epochs"] + 1):
        
        # KL Annealing: gradually increase from 0 to target over first 500 epochs
        kl_weight = min(0.001, 0.001 * epoch / 500.0)
        
        # Train
        avg_loss, avg_recon, avg_kl = train_one_epoch(
            model, dataloader, optimizer, device, epoch, kl_weight=kl_weight
        )
        current_lr = optimizer.param_groups[0]['lr']
    
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss

            if epoch > 50:
                torch.save(model.state_dict(), "best_gaussian_vae.pth")
                print(f"--> New best model saved (Loss: {best_loss:.4f})")
            
        # Log
        wandb.log({
            "epoch": epoch, 
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "kl_weight": kl_weight,
            "learning_rate": current_lr,
            "best_loss": best_loss
        })
        print(f"Epoch {epoch}/{CONFIG['train']['max_epochs']} | "
              f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}) | "
              f"LR: {current_lr:.6f}")

        # Step LR Scheduler (once per epoch)
        lr_scheduler.step()
            
        # Periodic Sampling (sample from latent space)
        if epoch % SAMPLE_SAVE_RATE == 0:
            sample_from_latent(model, device, num_samples=5, epoch=epoch)
            
        # Periodic Save
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    # Final Wrap up
    torch.save(model.state_dict(), "final_vae_model.pth")
    print("Training Complete. Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
