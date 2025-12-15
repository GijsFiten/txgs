import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math
import os
from tqdm import tqdm  # Recommended for progress bars

from utils.dataset_helper import create_dataloaders
from utils.diffusion_data_helper import DiffusionScheduler, denormalize_data
from diffusion_model import GaussianDiffusionTransformer
from utils.image_utils import render_and_save

# --- Configuration ---
CONFIG = {
    "data_dir": "./data/chairs_1k/",
    "output_dir": "./output/",
    "batch_size": 16,      
    "grad_accumulation": 3, 
    "model": {
        "input_dim": 8,
        "model_dim": 512,
        "n_heads": 8,
        "n_layers": 8,
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


# --- Core: Sampling Function ---
def sample_and_render(model, scheduler, device, num_samples=5, epoch=None):
    """Sample gaussians from the model and render them."""
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    with torch.no_grad():
        input_dim = CONFIG["model"]["input_dim"]
        # Start from pure noise
        x = torch.randn(num_samples, 1000, input_dim, device=device)
        
        # tqdm for sampling progress
        iterator = tqdm(reversed(range(scheduler.num_timesteps)), desc="Sampling", leave=False)
        
        for t in iterator:
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, timesteps)
            
            # Scheduler constants
            alpha_t = scheduler.alphas[t].to(device)
            alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)
            beta_t = scheduler.betas[t].to(device)
            
            # Denoising Step
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            
            x = coef1 * (x - coef2 * predicted_noise)
            x = torch.clamp(x, -3.5, 3.5) # Dynamic range clipping
            
            # Add noise (Langevin dynamics) except at last step
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x += sigma_t * noise
        
        # Render
        print(f"\n[Rendering] Saving {num_samples} samples...")
        for i in range(num_samples):
            sample = x[i]
            # Denormalize
            xy, scale, rot, feat = denormalize_data(
                sample[:, 0:2], sample[:, 2:4], sample[:, 4:5], sample[:, 5:8]
            )
            
            # Ensure all tensors are float32 and contiguous
            xy = xy.contiguous().float()
            scale = scale.contiguous().float()
            rot = rot.contiguous().float()
            feat = feat.contiguous().float()
            
            # Ensure img_size is explicitly int tuple
            img_size = (int(480), int(640))
            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
            filename = f"{CONFIG['output_dir']}/sample_{i}{epoch_suffix}"
            
            render_and_save(xy, scale, rot, feat, filename, img_size)
    
    model.train()


# --- Core: Training Loop ---
def train_one_epoch(model, dataloader, optimizer, diffusion_scheduler, device, epoch):
    model.train()
    epoch_loss = 0
    valid_batches = 0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    optimizer.zero_grad() # Reset gradients at start of epoch
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        B = batch.shape[0]

        # 1. Safety Checks
        if torch.isnan(batch).any():
            print(f"Warning: NaN detected in input batch at index {batch_idx}, skipping this batch.")
            continue

        # 2. Diffusion Process
        t = torch.randint(0, diffusion_scheduler.num_timesteps, (B,), device=device)
        noise = torch.randn_like(batch)
        noisy_samples = diffusion_scheduler.add_noise(batch, noise, t)

        # 3. Prediction
        predicted_noise = model(noisy_samples, t)
        
        # 4. Loss Computation
        loss = model.compute_loss(predicted_noise, noise)
        
        if torch.isnan(loss):
            continue

        # 5. Gradient Accumulation
        # Normalize loss by accumulation steps so gradients sum correctly
        norm_loss = loss / CONFIG["grad_accumulation"]
        norm_loss.backward()

        # 6. Optimization Step (only every N steps)
        if (batch_idx + 1) % CONFIG["grad_accumulation"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["train"]["clip_norm"])
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        current_loss = loss.item()
        epoch_loss += current_loss
        valid_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")

    # 1. Data
    dataloader = create_dataloaders(CONFIG["data_dir"], batch_size=CONFIG["batch_size"])

    # 2. Model (ensure you added _init_weights in the class as discussed!)
    model = GaussianDiffusionTransformer(
        input_dim=CONFIG["model"]["input_dim"], 
        model_dim=CONFIG["model"]["model_dim"], 
        n_heads=CONFIG["model"]["n_heads"], 
        n_layers=CONFIG["model"]["n_layers"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params / 1e6:.2f}M Params")

    # 3. Optimization Components
    # Using AdamW for better weight decay handling in Transformers
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["train"]["base_lr"], weight_decay=1e-4)
    
    # Custom Warmup Scheduler
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=CONFIG["train"]["warmup_epochs"], 
        max_epochs=CONFIG["train"]["max_epochs"]
    )

    diffusion_scheduler = DiffusionScheduler(num_timesteps=CONFIG["diffusion_steps"])
    # Move scheduler buffers to device manually if needed, or rely on .to(device) in loop
    diffusion_scheduler.sqrt_alphas_cumprod = diffusion_scheduler.sqrt_alphas_cumprod.to(device)
    diffusion_scheduler.sqrt_one_minus_alphas_cumprod = diffusion_scheduler.sqrt_one_minus_alphas_cumprod.to(device)

    # 4. Logging
    wandb.init(project="gaussian-diffusion", config=CONFIG)
    best_loss = float('inf')

    # --- Main Loop ---
    for epoch in range(1, CONFIG["train"]["max_epochs"] + 1):
        
        # Train
        avg_loss = train_one_epoch(model, dataloader, optimizer, diffusion_scheduler, device, epoch)
        current_lr = optimizer.param_groups[0]['lr']
    
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss

            if epoch > 50:
                torch.save(model.state_dict(), "best_gaussian_diffusion.pth")
                print(f"--> New best model saved (Loss: {best_loss:.4f})")
            
        # Log
        wandb.log({
            "epoch": epoch, 
            "loss": avg_loss, 
            "learning_rate": current_lr,
            "best_loss": best_loss
        })
        print(f"Epoch {epoch}/{CONFIG['train']['max_epochs']} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

         # Step LR Scheduler (once per epoch)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
            
        # Periodic Sampling
        if epoch % SAMPLE_SAVE_RATE == 0:
            sample_and_render(model, diffusion_scheduler, device, num_samples=3, epoch=epoch)
            
        # Periodic Save
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    # Final Wrap up
    torch.save(model.state_dict(), "final_model.pth")
    print("Training Complete. Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
