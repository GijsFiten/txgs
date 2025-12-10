import numpy as np
import glob
import os

files = glob.glob(os.path.join('./data/', '*.npz'))
for fpath in files:
    data = np.load(fpath)
    
    xy_shape = data['xy'].shape
    scale_shape = data['scale'].shape
    rot_shape = data['rot'].shape
    feat_shape = data['feat'].shape

    #XY should be N x 2
    if xy_shape[1] != 2 or scale_shape[1] != 2 or rot_shape[1] != 1 or feat_shape[1] != 3:
        #delete the file
        os.remove(fpath)
        print(f"Deleted file {fpath} due to incorrect shape.")
        continue

    # if N is 10000; move the file to data/large/
    if xy_shape[0] == 10000:
        new_dir = './data/large/'
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, os.path.basename(fpath))
        # if the file already exists in the new location, add a suffix to avoid overwriting
        if os.path.exists(new_path):
            base, ext = os.path.splitext(os.path.basename(fpath))
            new_path = os.path.join(new_dir, f"{base}_1{ext}")
        os.rename(fpath, new_path)
        print(f"Moved large file {fpath} to {new_path}")

    if xy_shape[0] == 1000:
        new_dir = './data/small/'
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, os.path.basename(fpath))
        # if the file already exists in the new location, add a suffix to avoid overwriting
        if os.path.exists(new_path):
            base, ext = os.path.splitext(os.path.basename(fpath))
            new_path = os.path.join(new_dir, f"{base}_1{ext}")
        os.rename(fpath, new_path)
        print(f"Moved small file {fpath} to {new_path}")