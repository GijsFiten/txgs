import numpy as np
import torch

def load_gaussians(npz_path, device="cuda:0"):
    """
    Load gaussian parameters from an .npz file.
    
    Args:
        npz_path: Path to the .npz file
        device: Device to load the tensors onto ('cuda:0', 'cpu', etc.)
    Returns:
        dict: Dictionary containing loaded tensors and metadata
    """

    data = np.load(npz_path, allow_pickle=True)
    
    # Extract gaussian parameters
    xy = torch.from_numpy(data["xy"]).to(dtype=torch.float32, device=device)            # First two columns are xy
    scale = torch.from_numpy(data["scale"]).to(dtype=torch.float32, device=device)      # Next two columns are scale  # Convert from full width/height to stddev
    rot = torch.from_numpy(data["rot"]).to(dtype=torch.float32, device=device)          # Next column is rotation
    feat = torch.from_numpy(data["feat"]).to(dtype=torch.float32, device=device)        # Remaining columns are colour features
    # Let's clip features to [0, 1]
    feat = torch.clamp(feat, 0.0, 1.0)
    
    # Extract metadata
    img_h = int(data["img_h"])
    img_w = int(data["img_w"])
    feat_dim = int(data["feat_dim"])
    input_channels = data["input_channels"].tolist() if "input_channels" in data else [feat_dim]
    disable_inverse_scale = bool(data["disable_inverse_scale"]) if "disable_inverse_scale" in data else False
    disable_topk_norm = bool(data["disable_topk_norm"]) if "disable_topk_norm" in data else False
    disable_tiles = bool(data["disable_tiles"]) if "disable_tiles" in data else False
    topk = int(data["topk"]) if "topk" in data else 10
    gamma = float(data["gamma"]) if "gamma" in data else 1.0

    scale = torch.abs(scale)

    if not disable_inverse_scale:
        scale = 1.0 / scale
    
    return dict(
        xy=xy,
        scale=scale,
        rot=rot,
        feat=feat,
        img_h=img_h,
        img_w=img_w,
        feat_dim=feat_dim,
        input_channels=input_channels,
        disable_inverse_scale=disable_inverse_scale,
        disable_topk_norm=disable_topk_norm,
        disable_tiles=disable_tiles,
        topk=topk,
        gamma=gamma
    )

def save_gaussians(gaussian_tensor: torch.Tensor, config, path):
        gaussians_data = {
            "xy": gaussian_tensor[:,0:2].detach().clone().cpu().numpy(),        # First two columns are xy
            "scale": gaussian_tensor[:,2:4].detach().clone().cpu().numpy(),     # Next two columns are scale
            "rot": gaussian_tensor[:,4:5].detach().clone().cpu().numpy(),         # Next column is rotation
            "feat": gaussian_tensor[:,5:8].detach().clone().cpu().numpy(),      # Remaining columns are colour features
            # Save metadata needed for rendering
            "img_h": config.img_h,
            "img_w": config.img_w,
            "num_gaussians": config.num_gaussians,
            "feat_dim": config.feat_dim,
            "input_channels": config.input_channels,
            "disable_inverse_scale": config.disable_inverse_scale,
            "disable_topk_norm": config.disable_topk_norm,
            "disable_tiles": config.disable_tiles,
            "topk": config.topk,
            "gamma": config.gamma,
            # Save bit precision info
            "quantize": config.quantize,
            "pos_bits": config.pos_bits,
            "scale_bits": config.scale_bits,
            "rot_bits": config.rot_bits,
            "feat_bits": config.feat_bits,
        }
        np.savez_compressed(path, **gaussians_data)