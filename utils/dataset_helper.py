import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import glob
import os

from .diffusion_data_helper import normalize_data
from .gaussian_file_helper import load_gaussians

class GaussianSplatDataset(Dataset):
    def __init__(self, data_dir, device="cpu", augment=False, file_paths=None):
        """
        Args:
            data_dir (str): Path to folder containing .npz files
            device (str): Device for temporary loading ('cpu' recommended for dataloader)
            augment (bool): Whether to apply data augmentation
            file_paths (list): Optional list of file paths. If provided, ignores data_dir glob.
        """
        if file_paths is not None:
            self.file_paths = file_paths
        else:
            self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
            
        self.device = device
        self.augment = augment
        
        if len(self.file_paths) == 0:
            print(f"Warning: No .npz files found in {data_dir}")
        else:
            if file_paths is None:
                print(f"Found {len(self.file_paths)} files in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load using the existing gaussian_file_helper
        path = self.file_paths[idx]
        try:
            data = load_gaussians(path, device="cpu")

            xy = torch.Tensor(data["xy"])
            scale = torch.Tensor(data["scale"])
            rot = torch.Tensor(data["rot"])
            feat = torch.Tensor(data["feat"])
            
            # Check for NaN or Inf in raw data
            if torch.isnan(xy).any() or torch.isinf(xy).any():
                raise ValueError(f"NaN/Inf in xy")
            if torch.isnan(scale).any() or torch.isinf(scale).any():
                raise ValueError(f"NaN/Inf in scale")
            if torch.isnan(rot).any() or torch.isinf(rot).any():
                raise ValueError(f"NaN/Inf in rot")
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                raise ValueError(f"NaN/Inf in feat")
            
            # Handle grayscale
            if feat.shape[1] == 1:
                feat = feat.expand(-1, 3)
            
            # Apply augmentation BEFORE normalization
            if self.augment:
                xy, scale, rot, feat = self.augment_gaussians(xy, scale, rot, feat)
            
            # Normalize the data
            xy_n, scale_n, rot_n, feat_n = normalize_data(xy, scale, rot, feat)
            
            # Sort by scale norm
            scale_norms = torch.norm(scale_n, dim=1)
            sorted_indices = torch.argsort(scale_norms, descending=True)
            
            xy_n = xy_n[sorted_indices]
            scale_n = scale_n[sorted_indices]
            rot_n = rot_n[sorted_indices]
            feat_n = feat_n[sorted_indices]
            
            # Check for NaN after normalization
            if torch.isnan(xy_n).any() or torch.isnan(scale_n).any() or torch.isnan(rot_n).any() or torch.isnan(feat_n).any():
                raise ValueError(f"NaN after normalization")
            
            # Concatenate all normalized features
            x_0 = torch.cat([xy_n, scale_n, rot_n, feat_n], dim=-1)

            return x_0

        except Exception as e:
            # print(f"Error loading {path}: {e}")
            # Recursive fallback to next item
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
    
    def augment_gaussians(self, xy, scale, rot, feat):
        """
        Apply data augmentation to gaussians (before normalization)
        All operations are differentiable and reversible
        """
        # 1. Horizontal Flip (50% chance)
        if torch.rand(1) < 0.5:
            xy[:, 0] = 1.0 - xy[:, 0]  # Flip x-coordinate (assuming normalized to [0,1])
            rot = -rot  # Flip rotation angle
        
        # 2. Vertical Flip (50% chance)
        if torch.rand(1) < 0.5:
            xy[:, 1] = 1.0 - xy[:, 1]  # Flip y-coordinate
            rot = np.pi - rot  # Adjust rotation
        
        # 3. 90-degree rotations (25% chance each: 0°, 90°, 180°, 270°)
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:  # 90° clockwise
            xy_new = xy.clone()
            xy_new[:, 0] = xy[:, 1]
            xy_new[:, 1] = 1.0 - xy[:, 0]
            xy = xy_new
            rot = rot - np.pi / 2
            # Swap scale x/y for 90° rotation
            scale = torch.flip(scale, dims=[1])
            
        elif rotation_choice == 2:  # 180°
            xy = 1.0 - xy
            rot = rot + np.pi
            
        elif rotation_choice == 3:  # 270° clockwise (90° counter-clockwise)
            xy_new = xy.clone()
            xy_new[:, 0] = 1.0 - xy[:, 1]
            xy_new[:, 1] = xy[:, 0]
            xy = xy_new
            rot = rot + np.pi / 2
            # Swap scale x/y for 270° rotation
            scale = torch.flip(scale, dims=[1])
        
        # 4. Color jitter (cheap and effective)
        if torch.rand(1) < 0.7:  # 70% chance
            # Brightness shift
            brightness = 0.9 + torch.rand(1) * 0.2  # [0.9, 1.1]
            feat = feat * brightness
            
            # Contrast
            contrast = 0.9 + torch.rand(1) * 0.2  # [0.9, 1.1]
            feat_mean = feat.mean(dim=0, keepdim=True)
            feat = feat_mean + (feat - feat_mean) * contrast
            
            # Saturation (for RGB)
            if feat.shape[1] == 3:
                gray = feat.mean(dim=1, keepdim=True)
                saturation = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
                feat = gray + (feat - gray) * saturation
            
            # Clamp to valid range
            feat = torch.clamp(feat, 0, 1)
        
        # 5. Scale perturbation (small random scaling)
        if torch.rand(1) < 0.5:
            scale_factor = 0.9 + torch.rand(1) * 0.2  # [0.9, 1.1]
            scale = scale * scale_factor
        
        # 6. Random translation (small jitter)
        if torch.rand(1) < 0.5:
            translate_x = (torch.rand(1) - 0.5) * 0.1  # ±5%
            translate_y = (torch.rand(1) - 0.5) * 0.1
            xy[:, 0] = torch.clamp(xy[:, 0] + translate_x, 0, 1)
            xy[:, 1] = torch.clamp(xy[:, 1] + translate_y, 0, 1)
        
        # Normalize rotation to [-π, π]
        rot = torch.atan2(torch.sin(rot), torch.cos(rot))
        return xy, scale, rot, feat

def create_dataloaders(data_dir, batch_size=32, num_points=1000, shuffle=True, augment=False, is_distributed=False):
    
    # Create Dataset
    dataset = GaussianSplatDataset(
        data_dir=data_dir,
        augment=augment
    )

    sampler = None
    if is_distributed:
        # 1. Create the DistributedSampler
        # It handles shuffling automatically if shuffle=True
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        # 2. If distributed, we MUST turn off loader-level shuffling
        # because the sampler handles it.
        shuffle=(shuffle and not is_distributed), 
        sampler=sampler, 
        num_workers=4,
        pin_memory=True, # Recommended True for CUDA
        drop_last=False
    )
    
    # 3. Return the sampler too! You need it for set_epoch() in the training loop.
    return dataloader, sampler


def create_train_val_dataloaders(data_dir, batch_size=32, validation_split=0.05, shuffle=True, augment=True, is_distributed=False):
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # Deterministic shuffle for splitting
    # We sort first to ensure order before shuffle across different processes, then use fixed seed
    all_files.sort()
    rng = np.random.RandomState(42)
    rng.shuffle(all_files)
    
    val_size = int(len(all_files) * validation_split)
    # Ensure at least one validation file if dataset is large enough
    if val_size == 0 and len(all_files) > 0:
         val_size = 1
    
    # Check if dataset is empty
    if len(all_files) == 0:
        print(f"Warning: No files found in {data_dir}")
        train_files = []
        val_files = []
    else:
        train_files = all_files[val_size:]
        val_files = all_files[:val_size]
    
    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation files")

    # Train dataset (with augmentation)
    train_dataset = GaussianSplatDataset(data_dir, augment=augment, file_paths=train_files) # Augmentation can be turned on later if needed
    
    # Val dataset (no augmentation)
    val_dataset = GaussianSplatDataset(data_dir, augment=False, file_paths=val_files)

    train_sampler = None
    val_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Train DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(shuffle and not is_distributed), 
        sampler=train_sampler, 
        num_workers=4,
        pin_memory=True,
        drop_last=True 
    )
    
    # Val DataLoader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        sampler=val_sampler, 
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler, val_sampler