from utils.dataset_helper import create_dataloaders
from utils.diffusion_data_helper import DiffusionScheduler, denormalize_data   
from model import GaussianDiffusionTransformer
import torch
import wandb

from utils.image_utils import render_and_save

def sample_and_render(model, diffusion_scheduler, device, num_samples=5, epoch=None):
    """Sample gaussians from the model and render them."""
    model.eval()
    with torch.no_grad():
        input_dim = 8
        # Start from pure noise
        sampled_gaussians = torch.randn(num_samples, 1000, input_dim, device=device)
        
        # Full denoising schedule (all 1000 steps)
        for t in reversed(range(diffusion_scheduler.num_timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(sampled_gaussians, timesteps)
            
            # Get scheduler parameters
            alpha_t = diffusion_scheduler.alphas[t].to(device)
            alpha_cumprod_t = diffusion_scheduler.alphas_cumprod[t].to(device)
            beta_t = diffusion_scheduler.betas[t].to(device)
            
            # Compute denoised sample
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            
            sampled_gaussians = coef1 * (sampled_gaussians - coef2 * predicted_noise)
            sampled_gaussians = torch.clamp(sampled_gaussians, -3.5, 3.5)
            
            # Add noise for all steps except the last
            if t > 0:
                noise = torch.randn_like(sampled_gaussians)
                sigma_t = torch.sqrt(beta_t)
                sampled_gaussians += sigma_t * noise
            
            if t % 100 == 0:
                print(f"Denoising step {t}/1000")
        
        # Denormalize before rendering
        for i in range(num_samples):
            sample = sampled_gaussians[i]  # [1000, 8]
            
            # Split into components
            xy_norm = sample[:, 0:2]
            scale_norm = sample[:, 2:4]
            rot_norm = sample[:, 4:5]
            feat_norm = sample[:, 5:8]
            
            # Denormalize
            xy, scale, rot, feat = denormalize_data(xy_norm, scale_norm, rot_norm, feat_norm)
            
            # Ensure all tensors are float32 and contiguous
            xy = xy.contiguous().float()
            scale = scale.contiguous().float()
            rot = rot.contiguous().float()
            feat = feat.contiguous().float()
            
            print(f"Sample {i} - xy: [{xy.min():.3f}, {xy.max():.3f}], "
                  f"scale: [{scale.min():.3f}, {scale.max():.3f}], "
                  f"rot: [{rot.min():.3f}, {rot.max():.3f}], "
                  f"feat: [{feat.min():.3f}, {feat.max():.3f}]")
            
            # Render with proper image size
            img_size = (int(480), int(640))  # (height, width)
            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
            render_and_save(xy, scale, rot, feat, f"output/sample_{i}{epoch_suffix}", img_size)
    
    model.train()

def main():
    data_dir = "./data/small/"  # Path to your .npz files
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoader
    dataloader = create_dataloaders(data_dir, batch_size=batch_size)

    # Initialize Model
    # Example: XY(2) + Scale(2) + Rot(1) + Feat(3) = 8 Channels
    input_dim = 8
    model = GaussianDiffusionTransformer(input_dim=input_dim, model_dim=512, n_heads=8, n_layers=12).to(device)

    print("Model initialized.")
    print(f"Using device: {device}")
    #param count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Initialize wandb
    wandb.init(
        project="gaussian-diffusion",
        config={
            "batch_size": batch_size,
            "input_dim": input_dim,
            "model_dim": 256,
            "n_heads": 8,
            "n_layers": 6,
            "total_params": total_params,
            "max_epochs": 5000,
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealing",
            "diffusion_timesteps": 1000,
        }
    )

    # Initialize Diffusion Scheduler
    diffusion_scheduler = DiffusionScheduler(num_timesteps=1000)
    diffusion_scheduler.sqrt_alphas_cumprod = diffusion_scheduler.sqrt_alphas_cumprod.to(device)
    diffusion_scheduler.sqrt_one_minus_alphas_cumprod = diffusion_scheduler.sqrt_one_minus_alphas_cumprod.to(device)

    MAX_EPOCHS = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_loss = float('inf')

    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = batch.to(device)
            
            # batch: [B, N_Gaussians, Features]
            B, N, C = batch.shape
            assert N == 1000 and C == 8
            
            # Debug: Check for NaN in input data
            if torch.isnan(batch).any():
                print(f"WARNING: NaN found in input batch {batch_idx}")
                continue

            # Sample random timesteps for each sample in the batch
            timesteps = torch.randint(0, diffusion_scheduler.num_timesteps, (B,), device=device)

            # Sample noise
            noise = torch.randn_like(batch)

            # Add noise to the original samples
            noisy_samples = diffusion_scheduler.add_noise(batch, noise, timesteps)
            
            if torch.isnan(noisy_samples).any():
                print(f"WARNING: NaN in noisy_samples at batch {batch_idx}")
                print(f"Timesteps: {timesteps}")
                print(f"Batch stats: min={batch.min()}, max={batch.max()}, mean={batch.mean()}")
                continue

            # Predict the noise using the model
            predicted_noise = model(noisy_samples, timesteps)
            
            if torch.isnan(predicted_noise).any():
                print(f"WARNING: NaN in predicted_noise at batch {batch_idx}")
                print(f"Noisy samples stats: min={noisy_samples.min()}, max={noisy_samples.max()}")
                continue

            # Compute loss between predicted noise and ground truth noise
            loss = model.compute_loss(predicted_noise, noise)
            
            if torch.isnan(loss):
                print(f"WARNING: NaN loss at batch {batch_idx}")
                continue

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print first batch details
            if epoch == 0 and batch_idx == 0:
                print(f"First batch debug:")
                print(f"  Batch range: [{batch.min():.4f}, {batch.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}")
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if avg_loss < best_loss and epoch > 50:
                best_loss = avg_loss
                torch.save(model.state_dict(), "best_gaussian_diffusion_transformer.pth")
                print(f"New best model saved with loss {best_loss:.4f} at epoch {epoch+1}")
                wandb.log({"best_loss": best_loss})
            print(f"Epoch {epoch+1}/{MAX_EPOCHS}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{MAX_EPOCHS}, No valid batches!")
            wandb.log({"epoch": epoch + 1, "loss": float('nan')})
        
        # Sample and render every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"\n=== Sampling at epoch {epoch+1} ===")
            sample_and_render(model, diffusion_scheduler, device, num_samples=3, epoch=epoch+1)
            print("=== Sampling complete ===\n")
        
        # Let's save the model every 500 epochs
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), f"gaussian_diffusion_transformer_epoch{epoch+1}.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")
        
    
    # Training is done, let's save the model
    torch.save(model.state_dict(), "gaussian_diffusion_transformer.pth")
    print("Model saved as gaussian_diffusion_transformer.pth")

    # Final sampling after training
    print("\n=== Final sampling ===")
    sample_and_render(model, diffusion_scheduler, device, num_samples=5)
    
    # Finish wandb run
    wandb.finish()

#

if __name__ == "__main__":
    main()