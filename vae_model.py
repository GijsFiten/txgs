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

class GaussianVAE(nn.Module):
    def __init__(self, num_gaussians=1000, input_dim=8, latent_dim=256):  # Reduced from 512
        super(GaussianVAE, self).__init__()
        self.num_gaussians = num_gaussians
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Smaller PointNet++ architecture
        # sa1: 1000 -> 256 points (reduced from 512)
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=256,  # Reduced
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[8, 16, 32],  # Reduced from [16, 32, 64]
            in_channel=6,
            mlp_list=[[16, 16, 32], [32, 32, 64], [32, 48, 64]]  # Smaller channels
        )
        # sa1 output: 32 + 64 + 64 = 160 channels (vs 320 before)
        
        # sa2: 256 -> 64 points (reduced from 128)
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=64,  # Reduced
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[16, 32, 64],  # Keep same
            in_channel=160,  # Updated
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 64, 128]]  # Smaller
        )
        # sa2 output: 64 + 128 + 128 = 320 channels (vs 640 before)
        
        # sa3: 64 -> 1 global feature (group_all=True)
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=320 + 2,  # Updated: sa2 output (320) + xy (2)
            mlp=[128, 256, 512],  # Reduced from [256, 512, 1024]
            group_all=True
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(512, latent_dim)  # 512 from sa3 output
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder: Smaller MLP
        self.decoder_fc1 = nn.Linear(latent_dim, 256)  # Reduced from 512
        self.decoder_bn1 = nn.BatchNorm1d(256)
        self.decoder_fc2 = nn.Linear(256, 512)  # Reduced from 1024
        self.decoder_bn2 = nn.BatchNorm1d(512)
        self.decoder_fc3 = nn.Linear(512, num_gaussians * input_dim)
        
    def encode(self, x):
        """
        Encode gaussians to latent space
        Args:
            x: [B, N, 8] - gaussian parameters
        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        B, N, C = x.shape
        
        # Split into positions (xy) and features (scale, rot, feat)
        xy = x[:, :, :2].permute(0, 2, 1)  # [B, 2, N] - positions
        features = x[:, :, 2:].permute(0, 2, 1)  # [B, 6, N] - other features
        
        # PointNet++ encoder
        l1_xyz, l1_points = self.sa1(xy, features)  # [B, 2, 512], [B, C1, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 2, 128], [B, C2, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 2, 1], [B, 1024, 1]
        
        # Global feature vector
        global_feature = l3_points.view(B, -1)  # [B, 1024]
        
        # VAE latent parameters
        mu = self.fc_mu(global_feature)  # [B, latent_dim]
        logvar = self.fc_logvar(global_feature)  # [B, latent_dim]
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to gaussians
        Args:
            z: [B, latent_dim]
        Returns:
            x_recon: [B, N, 8] - reconstructed gaussians
        """
        B = z.shape[0]
        
        # MLP decoder
        x = F.relu(self.decoder_bn1(self.decoder_fc1(z)))
        x = F.relu(self.decoder_bn2(self.decoder_fc2(x)))
        x = self.decoder_fc3(x)  # [B, N*8]
        
        # Reshape to gaussian parameters
        x_recon = x.view(B, self.num_gaussians, self.input_dim)  # [B, N, 8]
        
        # Clamp to reasonable normalized range
        x_recon = torch.clamp(x_recon, -3.5, 3.5)
        
        return x_recon
    
    def forward(self, x):
        """
        Full VAE forward pass
        Args:
            x: [B, N, 8] - input gaussians
        Returns:
            x_recon: [B, N, 8] - reconstructed gaussians
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    

class GaussianVAETiny(nn.Module):
    def __init__(self, num_gaussians=1000, input_dim=8, latent_dim=128):
        super(GaussianVAETiny, self).__init__()
        self.num_gaussians = num_gaussians
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Super minimal encoder
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.15, 0.3],  # Only 2 scales
            nsample_list=[16, 32],
            in_channel=6,
            mlp_list=[[16, 32], [32, 64]]  # Minimal depth
        )
        # Output: 32 + 64 = 96 channels
        
        # Global pooling only
        self.sa2 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=96 + 2,
            mlp=[64, 128, 256],
            group_all=True
        )
        
        # Latent
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Tiny decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_gaussians * input_dim)
        )
    
    def encode(self, x):
        B, N, C = x.shape
        xy = x[:, :, :2].permute(0, 2, 1)
        features = x[:, :, 2:].permute(0, 2, 1)
        
        l1_xyz, l1_points = self.sa1(xy, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        global_feature = l2_points.view(B, -1)
        mu = self.fc_mu(global_feature)
        logvar = self.fc_logvar(global_feature)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        x_recon = x.view(-1, self.num_gaussians, self.input_dim)
        return torch.clamp(x_recon, -3.5, 3.5)


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