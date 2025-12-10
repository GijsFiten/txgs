import torch
import math

def normalize_data(xy, scale, rot, feat):
    xy_norm = (xy - 0.5) / 0.2
    # Clamp scale to prevent log of non-positive numbers
    scale_clamped = torch.clamp(scale, min=1e-6)
    scale_norm = (torch.log(scale_clamped) - 3) / 2
    rot_norm = rot * 2.0
    # Clamp feat to prevent extreme values
    feat_clamped = torch.clamp(feat, min=0.0, max=1.0)
    feat_norm = (feat_clamped * 2.0) - 1.0
    return xy_norm, scale_norm, rot_norm, feat_norm

def denormalize_data(xy_norm, scale_norm, rot_norm, feat_norm):
    xy = xy_norm * 0.2 + 0.5
    scale = torch.exp(scale_norm * 2 + 3)
    rot = rot_norm / 2.0
    feat = (feat_norm + 1.0) / 2.0
    feat = torch.clamp(feat, 0.0, 1.0)
    return xy, scale, rot, feat


class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule="cosine"):
        self.num_timesteps = num_timesteps
        self.schedule = schedule

        if schedule == "linear":
            # Standard DDPM schedule
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            # Better for high-res/detailed data (prevents early signal destruction)
            self.betas = self._cosine_beta_schedule(num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For the forward process (q_sample)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward diffusion process: q(x_t | x_0)
        original_samples: [B, C, N] (Your normalized gaussians)
        noise: [B, C, N] (Gaussian noise)
        timesteps: [B] (Integer timesteps)
        """
        # Broadcast mechanism to match shape
        # We need to extract the specific alpha for each item in the batch
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape to [Batch, 1, 1] for broadcasting against [Batch, Channels, Points]
        s1 = s1.view(-1, *([1] * (original_samples.ndim - 1)))
        s2 = s2.view(-1, *([1] * (original_samples.ndim - 1)))
        
        noisy_samples = s1 * original_samples + s2 * noise
        return noisy_samples

    def sample_timesteps(self, batch_size, device):
        """Get random timesteps for training"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()