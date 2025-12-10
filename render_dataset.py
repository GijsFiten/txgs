import os
import glob
import torch
from tqdm import tqdm
from utils.gaussian_file_helper import load_gaussians
from utils.image_utils import render_and_save

def render_dataset(data_dir, output_dir, device="cuda"):
    """
    Render all .npz files in a directory to verify dataset validity.
    
    Args:
        data_dir: Directory containing .npz files
        output_dir: Directory to save rendered images
        device: Device to use for rendering
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all npz files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return
    
    print(f"Found {len(npz_files)} files to render")
    print(f"Output directory: {output_dir}")
    print(f"Using device: {device}\n")
    
    failed_files = []
    
    # Render each file
    for npz_path in tqdm(npz_files, desc="Rendering dataset"):
        try:
            # Get filename without extension for output
            basename = os.path.basename(npz_path)
            filename = os.path.splitext(basename)[0]
            
            # Load gaussians
            data = load_gaussians(npz_path, device=device)
            
            xy = data["xy"]
            scale = data["scale"]
            rot = data["rot"]
            feat = data["feat"]
            img_h = data["img_h"]
            img_w = data["img_w"]
            
            # Check for invalid values
            if torch.isnan(xy).any() or torch.isinf(xy).any():
                print(f"\n⚠️  NaN/Inf in xy: {basename}")
                failed_files.append((basename, "NaN/Inf in xy"))
                continue
                
            if torch.isnan(scale).any() or torch.isinf(scale).any():
                print(f"\n⚠️  NaN/Inf in scale: {basename}")
                failed_files.append((basename, "NaN/Inf in scale"))
                continue
                
            if torch.isnan(rot).any() or torch.isinf(rot).any():
                print(f"\n⚠️  NaN/Inf in rot: {basename}")
                failed_files.append((basename, "NaN/Inf in rot"))
                continue
                
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                print(f"\n⚠️  NaN/Inf in feat: {basename}")
                failed_files.append((basename, "NaN/Inf in feat"))
                continue
            
            # Render
            img_size = (int(img_h), int(img_w))
            output_path = os.path.join(output_dir, filename)
            render_and_save(xy, scale, rot, feat, output_path, img_size)
            
        except Exception as e:
            print(f"\n❌ Error rendering {basename}: {str(e)}")
            failed_files.append((basename, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print(f"Rendering complete!")
    print(f"Successfully rendered: {len(npz_files) - len(failed_files)}/{len(npz_files)}")
    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    else:
        print("\n✅ All files rendered successfully!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render all gaussians in a dataset directory")
    parser.add_argument("--data_dir", type=str, default="./data/", 
                        help="Directory containing .npz files")
    parser.add_argument("--output_dir", type=str, default="./output/dataset_render/",
                        help="Directory to save rendered images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for rendering")
    
    args = parser.parse_args()
    
    render_dataset(args.data_dir, args.output_dir, args.device)
