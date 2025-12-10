import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

from .diffusion_data_helper import normalize_data
from .gaussian_file_helper import load_gaussians

class GaussianSplatDataset(Dataset):
    def __init__(self, data_dir, device="cpu"):
        """
        Args:
            data_dir (str): Path to folder containing .npz files
            device (str): Device for temporary loading ('cpu' recommended for dataloader)
        """
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        self.device = device
        
        if len(self.file_paths) == 0:
            print(f"Warning: No .npz files found in {data_dir}")
        else:
            print(f"Found {len(self.file_paths)} files in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load using the existing gaussian_file_helper which handles scale properly
        path = self.file_paths[idx]
        try:
            # Load on CPU first (dataloader will handle batching and device transfer)
            data = load_gaussians(path, device="cpu")

            xy = torch.Tensor(data["xy"])
            scale = torch.Tensor(data["scale"])
            rot = torch.Tensor(data["rot"])
            feat = torch.Tensor(data["feat"])
            
            # Check for NaN or Inf in raw data
            if torch.isnan(xy).any() or torch.isinf(xy).any():
                print(f"Warning: NaN/Inf in xy for {path}")
                return torch.zeros(1000, 8)
            if torch.isnan(scale).any() or torch.isinf(scale).any():
                print(f"Warning: NaN/Inf in scale for {path}")
                return torch.zeros(1000, 8)
            if torch.isnan(rot).any() or torch.isinf(rot).any():
                print(f"Warning: NaN/Inf in rot for {path}")
                return torch.zeros(1000, 8)
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                print(f"Warning: NaN/Inf in feat for {path}")
                return torch.zeros(1000, 8)
            
            # If we only have 1 colour features, its a grayscale image - repeat the feat channels
            if feat.shape[1] == 1:
                print(f"Info: Grayscale feat detected in {path}, repeating channels.")
                feat = feat.expand(-1, 3)

                
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1000, 8)
        
        # Normalize the data
        xy_n, scale_n, rot_n, feat_n = normalize_data(xy, scale, rot, feat)
        
        # Sort by scale norm (largest gaussians first for coarse-to-fine structure)
        scale_norms = torch.norm(scale_n, dim=1)  # L2 norm of [scale_x, scale_y]
        sorted_indices = torch.argsort(scale_norms, descending=True)  # Largest first
        
        # Reorder all features
        xy_n = xy_n[sorted_indices]
        scale_n = scale_n[sorted_indices]
        rot_n = rot_n[sorted_indices]
        feat_n = feat_n[sorted_indices]
        
        # Check for NaN after normalization
        if torch.isnan(xy_n).any() or torch.isnan(scale_n).any() or torch.isnan(rot_n).any() or torch.isnan(feat_n).any():
            print(f"Warning: NaN after normalization for {path}")
            print(f"  xy range: [{xy.min():.4f}, {xy.max():.4f}]")
            print(f"  scale range: [{scale.min():.4f}, {scale.max():.4f}]")
            print(f"  rot range: [{rot.min():.4f}, {rot.max():.4f}]")
            print(f"  feat range: [{feat.min():.4f}, {feat.max():.4f}]")
            return torch.zeros(1000, 8)
        
        # Concatenate all normalized features
        x_0 = torch.cat([xy_n, scale_n, rot_n, feat_n], dim=-1)

        return x_0

def create_dataloaders(data_dir, batch_size=32, num_points=1000):
    
    # Create Dataset
    dataset = GaussianSplatDataset(
        data_dir=data_dir
    )

    # Create DataLoader
    # num_workers=4 is usually a sweet spot for modern CPUs
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,       # Shuffle for training
        num_workers=4,      # Parallel loading
        pin_memory=True,    # Faster transfer to CUDA
        drop_last=True      # Drop incomplete batch at end
    )
    
    return dataloader