import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianDiffusionTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, n_heads=4, n_layers=6, max_num_gaussians=1000):
        super().__init__()
        
        # 1. Input Projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # 2. Geometric/Sequence Embedding (FIXED)
        # Instead of embedding the noisy XY coords, we learn a fixed embedding 
        # for each slot in the sequence (like BERT/ViT).
        self.pos_embedding = nn.Parameter(torch.randn(1, max_num_gaussians, model_dim))
        
        # 3. Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_heads, 
            dim_feedforward=model_dim*4, # Usually d_model * 4 is standard
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. Output Projection
        self.output_proj = nn.Linear(model_dim, input_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Standard Xavier initialization
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            # For your learned pos_embedding
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, x, t):
        # x: [Batch, N_Gaussians, Features]
        # t: [Batch]
        
        B, N, C = x.shape
        
        # A. Embed inputs
        h = self.input_proj(x)
        
        # B. Add Sequence Position Embedding (FIXED)
        # We add the learned embedding for positions 0..N
        h = h + self.pos_embedding[:, :N, :]
        
        # C. Add Time Embedding
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / 1000.0
        t_emb = self.time_mlp(t_normalized.unsqueeze(-1)).unsqueeze(1)
        
        # Add time info to every token
        h = h + t_emb
        
        # D. Process
        h = self.transformer(h)
        
        # E. Predict Noise
        return self.output_proj(h)
    
    def compute_loss(self, predicted_noise, ground_truth_noise):
        # Your loss function was actually fine, but simple MSE is often safer to start
        # Let's use different weights for different features if needed
        weights = torch.tensor([5.0, 5.0, 3.0, 3.0, 2.0, 0.5, 0.5, 0.5], device=predicted_noise.device)
        loss = F.mse_loss(predicted_noise * weights, ground_truth_noise * weights)
        return loss
