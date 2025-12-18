import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation
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
    def __init__(self, num_gaussians, latent_dim, output_dim=8, nhead=8, num_layers=4):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = latent_dim  # Directly use latent dimension
        
        # 1. Learnable "Query" Embeddings
        # These are the "empty slots" that will become your Gaussians.
        # Shape: [1, N, d_model]
        self.query_embeddings = nn.Parameter(torch.randn(1, num_gaussians, self.d_model))
        
        # 2. Standard Transformer Decoder
        # This architecture inherently supports Cross-Attention.
        # Queries (tgt) attend to Latent (memory).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4, 
            dropout=0.1, 
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN is crucial for convergence
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Output Heads
        self.head_xy = nn.Linear(self.d_model, 2)
        self.head_scale = nn.Linear(self.d_model, 2)
        self.head_rot = nn.Linear(self.d_model, 1)
        self.head_color = nn.Linear(self.d_model, 3)

    def forward(self, z):
        """
        Args:
            z: Latent vector [B, latent_dim]
        Returns:
            x: Gaussian parameters [B, N, 8]
        """
        B = z.shape[0]
        
        # A. Prepare Queries (Target)
        # These represent "Gaussian ID 0" to "Gaussian ID N"
        tgt = self.query_embeddings.expand(B, -1, -1) # [B, N, d_model]
        
        # B. Prepare Memory (Latent)
        # We treat the single latent vector as a sequence of length 1.
        # The decoder will Cross-Attend to this.
        memory = z.unsqueeze(1) # [B, 1, d_model]
        
        # C. Run Transformer Decoder
        # The magic happens here: 
        # 1. Self-Attention: Queries arrange themselves spatially.
        # 2. Cross-Attention: Queries look at 'memory' (z) to get shape info.
        x = self.decoder(tgt, memory)
        
        # D. Project to features
        xy = self.head_xy(x)
        scale = self.head_scale(x)
        rot = self.head_rot(x)
        color = self.head_color(x)
        
        return torch.cat([xy, scale, rot, color], dim=-1)

class GaussianVAE(nn.Module):
    def __init__(self, num_gaussians=1000, input_dim=8, latent_dim=768):
        super(GaussianVAE, self).__init__()
        self.num_gaussians = num_gaussians
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder (PointNet++) ---
        # (Your encoder code remains exactly the same)
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=256, radius_list=[0.1, 0.2, 0.4], nsample_list=[8, 16, 32],
            in_channel=6, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=64, radius_list=[0.2, 0.4, 0.8], nsample_list=[16, 32, 64],
            in_channel=320, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=640 + 2,
            mlp=[256, 512, 1024], group_all=True
        )
        
        # Projection to latent_dim (768)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # --- Decoder (Updated) ---
        self.decoder = GaussianTransformerDecoder(
            num_gaussians=num_gaussians,
            latent_dim=latent_dim, # Passes 768 directly
            output_dim=input_dim,
            nhead=8,     # 768 is divisible by 8 (96 dim per head)
            num_layers=4 
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
        # Run Decoder
        x_recon = self.decoder(z) 
        
        # Clamps to prevent Inf/NaNs in rasterizer
        xy = torch.clamp(x_recon[:, :, 0:2], -5.0, 5.0)
        
        # Hard clamp scale before exp to prevent overflow
        scale_norm = torch.clamp(x_recon[:, :, 2:4], max=5.0)
        
        rot = x_recon[:, :, 4:5]
        feat = x_recon[:, :, 5:8]
        
        return torch.cat([xy, scale_norm, rot, feat], dim=-1)
    
    def forward(self, x, use_sampling=True):
        mu, logvar = self.encode(x)
        if use_sampling:
            z = self.reparameterize(mu, logvar)
        else:
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

def vae_loss_sinkhorn(x_recon, x, mu, logvar, recon_weight=1.0, kl_weight=0.001, sinkhorn_epsilon=0.01):
    B, N, C = x.shape
    
    # 1. Compute Cost Matrix on ALL features (weighted)
    # Weight different features appropriately
    # XY: high weight (spatial is important)
    # Scale, Rotation: medium weight  
    # Color: lower weight
    
    pred_xy = x_recon[:, :, :2]
    gt_xy = x[:, :, :2]
    pred_scale = x_recon[:, :, 2:4]
    gt_scale = x[:, :, 2:4]
    pred_rot = x_recon[:, :, 4:5]
    gt_rot = x[:, :, 4:5]
    pred_color = x_recon[:, :, 5:8]
    gt_color = x[:, :, 5:8]
    
    # Compute per-feature costs [B, N, N]
    cost_xy = torch.sum((pred_xy.unsqueeze(2) - gt_xy.unsqueeze(1)) ** 2, dim=-1)
    cost_scale = torch.sum((pred_scale.unsqueeze(2) - gt_scale.unsqueeze(1)) ** 2, dim=-1)
    cost_color = torch.sum((pred_color.unsqueeze(2) - gt_color.unsqueeze(1)) ** 2, dim=-1)
    
    # Rotation: use angular distance (handles periodicity)
    rot_diff = pred_rot.unsqueeze(2) - gt_rot.unsqueeze(1)  # [B, N, N, 1]
    cost_rot = (1 - torch.cos(rot_diff)).squeeze(-1)  # [B, N, N]
    
    # Weighted total cost
    cost_matrix = 10.0 * cost_xy + 1.0 * cost_scale + 0.5 * cost_rot + 0.5 * cost_color
    
    # 2. Compute Soft Permutation Matrix (P) using Sinkhorn
    P = sinkhorn_matching(cost_matrix, epsilon=sinkhorn_epsilon, max_iter=50) # [B, N, N]
    
    # 3. Compute Transport Cost (Sinkhorn Loss)
    # Sum over N, N, average over B
    recon_loss = torch.sum(P * cost_matrix) / B

    
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