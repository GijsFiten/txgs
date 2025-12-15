import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg
from scipy.optimize import linear_sum_assignment


# This is a VAE model which takes in 2D Gaussian splats and encodes them into a latent space
# For now we always take in a fixed number of gaussians (e.g., 1000)
# Each gaussian has features: [xy (2), scale (2), rot (1), feat (3)] = 8 total features

# We build on pointnet++ style architecture for the encoder
# and a transformer based decoder to reconstruct the gaussians
# We use hungarian loss for matching gaussians between input and output
# Ideally we have a VAE that can compress the gaussian splat representation
# into a lower dimensional latent space for better diffusion modeling, using VAE to keep the
# latent space smooth and continuous

class GaussianTransformerDecoder(nn.Module):
    def __init__(self, num_gaussians, latent_dim, output_dim=8, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # 1. Learnable "Query" Embeddings
        # These act as the initial "slots" for the gaussians before they know what shape they are forming.
        # Shape: [1, N, d_model]
        self.query_embeddings = nn.Parameter(torch.randn(1, num_gaussians, d_model))
        
        # 2. Latent Projection
        # Projects your VAE latent (512) down to the transformer dim (128)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # 3. Transformer (Decoder-only style)
        # We use batch_first=True for easier handling
        # norm_first=True (Pre-LN) is crucial for training stability in generative transformers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Head
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, z):
        """
        Args:
            z: Latent vector [B, latent_dim]
        Returns:
            x: Gaussian parameters [B, N, 8]
        """
        B = z.shape[0]
        
        # A. Prepare Queries: [1, N, D] -> [B, N, D]
        queries = self.query_embeddings.expand(B, -1, -1)
        
        # B. Prepare Latent Condition: [B, latent_dim] -> [B, 1, D]
        z_emb = self.latent_proj(z).unsqueeze(1)
        
        # C. Condition the queries
        # We add the latent info to every query slot. This tells the slots *what* object to form.
        # The learnable query_embeddings tell the slots *which part* of the object they are responsible for.
        x = queries + z_emb
        
        # D. Run Transformer
        x = self.transformer(x)
        
        # E. Project to features
        x = self.output_head(x)
        
        return x

class GaussianVAE(nn.Module):
    def __init__(self, num_gaussians=1000, input_dim=8, latent_dim=512):
        super(GaussianVAE, self).__init__()
        self.num_gaussians = num_gaussians
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder (PointNet++) ---
        # Same as before
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[8, 16, 32],
            in_channel=6,
            mlp_list=[[16, 16, 32], [32, 32, 64], [32, 48, 64]]
        )
        
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[16, 32, 64],
            in_channel=160,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 64, 128]]
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=320 + 2,
            mlp=[128, 256, 512],
            group_all=True
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # --- Decoder (Transformer) ---
        self.decoder = GaussianTransformerDecoder(
            num_gaussians=num_gaussians,
            latent_dim=latent_dim,
            output_dim=input_dim, # 8
            d_model=128,          # Keep small
            nhead=4,              # 4 heads is sufficient for 128 dim
            num_layers=3          # 3 layers
        )
        
    def encode(self, x):
        B, N, C = x.shape
        xy = x[:, :, :2].permute(0, 2, 1)
        features = x[:, :, 2:].permute(0, 2, 1)
        
        l1_xyz, l1_points = self.sa1(xy, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        global_feature = l3_points.view(B, -1)
        mu = self.fc_mu(global_feature)
        logvar = self.fc_logvar(global_feature)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        B = z.shape[0]
        
        # Run Transformer Decoder
        x_recon = self.decoder(z) # [B, N, 8]
        
        # --- Post-Processing / Clamping ---
        
        # 1. XY: Tanh to keep in valid range (-1 to 1) roughly, or just clamp
        # Assuming normalized data is roughly -1 to 1.
        # We clamp to slightly larger to avoid boundary issues.
        xy = torch.clamp(x_recon[:, :, 0:2], -3.5, 3.5)
        
        # 2. Scale: Usually log-scale or small positive values.
        scale = x_recon[:, :, 2:4] 
        
        # 3. Rotation: Keep as is (raw radians)
        rot = x_recon[:, :, 4:5]
        
        # 4. Color (RGB): Sigmoid if 0-1, or Clamp if standardized
        # Assuming standardized input, so just clamp
        feat = x_recon[:, :, 5:8]
        
        return torch.cat([xy, scale, rot, feat], dim=-1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # OVERFIT
        z = mu
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
def vae_loss(x_recon, x, mu, logvar, recon_weight=1.0, kl_weight=0.001):
    """
    VAE loss = Reconstruction Loss (Hungarian matching) + KL Divergence
    
    Args:
        x_recon: [B, N, 8] - reconstructed gaussians
        x: [B, N, 8] - original gaussians
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
        recon_weight: weight for reconstruction loss
        kl_weight: weight for KL divergence (beta-VAE)
    """
    B, N, C = x.shape
    
    # Reconstruction loss with Hungarian matching
    recon_loss = 0.0
    for b in range(B):
        # Compute pairwise squared distances between all gaussians
        # [N, N] cost matrix where cost[i,j] = ||x_recon[i] - x[j]||^2
        x_recon_b = x_recon[b]  # [N, 8]
        x_b = x[b]  # [N, 8]
        
        # Compute cost matrix: squared L2 distance for each pair
        # Expand dims for broadcasting: [N, 1, 8] - [1, N, 8] = [N, N, 8]
        diff = x_recon_b.unsqueeze(1) - x_b.unsqueeze(0)
        cost_matrix = torch.sum(diff ** 2, dim=-1)  # [N, N]
        
        # Hungarian algorithm to find optimal assignment
        # Convert to numpy for scipy
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # Compute matched loss
        matched_loss = cost_matrix[row_ind, col_ind].sum()
        recon_loss += matched_loss
    
    # Average over batch
    recon_loss = recon_loss / B
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # Total loss
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

def vae_loss_sinkhorn(x_recon, x, mu, logvar, recon_weight=1.0, kl_weight=0.001, sinkhorn_epsilon=0.001):
    B, N, C = x.shape
    
    # 1. Compute Cost Matrix based on XY ONLY (Indices 0 and 1)
    # x_recon: [B, N, 8], x: [B, N, 8]
    pred_xy = x_recon[:, :, :2]
    gt_xy = x[:, :, :2]
    
    # Squared Euclidean Distance for XY
    # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N]
    dist_xy = torch.sum((pred_xy.unsqueeze(2) - gt_xy.unsqueeze(1)) ** 2, dim=-1)
    
    # 2. Compute Soft Permutation Matrix (P) using Sinkhorn
    # epsilon controls "hardness": 0.01 is sharper, 0.1 is softer/more stable
    P = sinkhorn_matching(dist_xy, epsilon=sinkhorn_epsilon, max_iter=50) # [B, N, N]
    
    # 3. Permute Ground Truth to match Prediction
    # We want x_matched[b, i] to be the point in x[b] that matches x_recon[b, i]
    # Matched GT = P * GT (Batch matrix multiplication)
    x_matched = torch.matmul(P, x) 
    
    # 4. Compute Loss on ALL features
    # Note: Since P is soft, x_matched is a weighted average of GT points.
    # As epsilon -> 0, it approaches a hard permutation.
    
    # Standard MSE for non-periodic features (XY, Scale, RGB)
    # Assuming indices: 0-2 (XY), 2-4 (Scale), 4-5 (Rot), 5-8 (RGB)
    
    # Loss for XY, Scale, RGB
    non_rot_mask = torch.ones(C, device=x.device, dtype=torch.bool)
    non_rot_mask[4] = False # Exclude rotation index
    
    loss_non_rot = F.mse_loss(x_recon[:, :, non_rot_mask], x_matched[:, :, non_rot_mask])
    
    # 5. Special Handling for Periodic Rotation (Index 4)
    # Since you want to keep 1D rotation, use Cosine distance for loss
    # to avoid 0 vs 2pi errors.
    rot_recon = x_recon[:, :, 4]
    rot_gt = x_matched[:, :, 4]
    
    # Loss = 1 - cos(pred - target)
    # This handles the modularity perfectly while keeping your 1D representation
    loss_rot = torch.mean(1 - torch.cos(rot_recon - rot_gt))
    
    # Total Reconstruction Loss
    recon_loss = loss_non_rot + 0.1 * loss_rot # Weight rotation as needed
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

def sinkhorn_matching(cost_matrix, epsilon=0.1, max_iter=50):
    """
    Numerically stable Sinkhorn-Knopp algorithm in log-space.
    
    Args:
        cost_matrix: [B, N, N] - Pairwise cost
        epsilon: Regularization parameter (higher = blurrier, lower = sharper)
        max_iter: Number of iterations
    """
    # Start with log(P) = -C / epsilon
    # We detach the cost matrix max to prevent extremely large negative numbers 
    # if costs explode, though log-space usually handles this well.
    log_P = -cost_matrix / epsilon

    for _ in range(max_iter):
        # 1. Row normalization in log space
        # log(P_norm) = log(P) - log(sum(exp(log_P)))
        # We use logsumexp which is numerically stable
        log_sum_rows = torch.logsumexp(log_P, dim=2, keepdim=True)
        log_P = log_P - log_sum_rows
        
        # 2. Column normalization in log space
        log_sum_cols = torch.logsumexp(log_P, dim=1, keepdim=True)
        log_P = log_P - log_sum_cols
        
    # Convert back to probability space for the final multiplication
    return torch.exp(log_P)