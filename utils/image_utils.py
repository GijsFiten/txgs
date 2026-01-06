import os

import cv2
import flip_evaluator
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from numpy.linalg import norm
from PIL import Image
from scipy.ndimage import sobel

FONT_PATH = "assets/fonts/linux_libertine/LinLibertine_R.ttf"
font_manager.fontManager.addfont(FONT_PATH)
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams['font.family'] = FONT_PROP
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['legend.title_fontsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

ALLOWED_IMAGE_FILE_FORMATS = [".jpeg", ".jpg", ".png", ".tiff", ".exr"]

PLOT_DPI = 72.0
GAUSSIAN_ZOOM = 5
GAUSSIAN_COLOR = "#80ed99"

from gsplat.gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_no_tiles, rasterize_gaussians_sum


def get_psnr(image1, image2, max_value=1.0):
    mse = torch.mean((image1-image2)**2)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr


def get_grid(h, w, x_lim=np.asarray([0, 1]), y_lim=np.asarray([0, 1])):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


def compute_image_gradients(image):
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))
        gx.append(sobel(image_channel, 1))
    gy = norm(np.stack(gy, axis=0), ord=2, axis=0).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=0).astype(np.float32)
    return gy, gx


def to_output_format(image, image_format, gamma):
    if image_format not in ALLOWED_IMAGE_FILE_FORMATS:
        raise ValueError(f"Invalid image format: {image_format}")
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Invalid image format: shape = {image.shape}")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
        if image.shape[2] not in [1, 3]:
            raise ValueError(f"Invalid image format: shape = {image.shape}")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    image = image.astype(np.float32)
    if gamma is not None:
        image = np.power(image, 1.0/gamma)
    if image_format in [".jpeg", ".jpg"]:
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image_format in [".png"]:
        image = np.clip(image, 0.0, 1.0)
        image = (65535.0 * image).astype(np.uint16)
    return image


def save_image(image, save_path, gamma=None, zoom=None):    
    image_format = os.path.splitext(save_path)[1].lower()
    image = to_output_format(image, image_format, gamma)
    if zoom is not None and zoom > 0.0:
        height, width = image.shape[:2]
        image = cv2.resize(image, (round(width*zoom), round(height*zoom)), interpolation=cv2.INTER_NEAREST)
    if len(image.shape) == 3:
        image = image[..., ::-1]
    cv2.imwrite(save_path, image)

def separate_and_save_images(images, channels, path):
        images_sep = separate_image_channels(images=images, input_channels=channels)
        for idx, image in enumerate(images_sep, 1):
            suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
            save_image(image, f"{path}{suffix}.{'png'}", gamma=1.0)


def separate_image_channels(images, input_channels):
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_list = []
    curr_channel = 0
    for num_channels in input_channels:
        image_list.append(images[curr_channel:curr_channel+num_channels])
        curr_channel += num_channels
    return image_list


def visualize_gaussian_footprint(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the footprint of Gaussians using colored elliptical disks.
    """
    if feat.shape[1] != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {feat.shape[1]:d} vs {sum(input_channels):d}")
    xy = xy.detach().cpu().clone().numpy()
    y, x = xy[:, 1] * img_h, xy[:, 0] * img_w
    scale = GAUSSIAN_ZOOM * scale.detach().cpu().clone().numpy()
    rot = rot.detach().cpu().clone().numpy()
    if gamma is not None:
        feat = torch.pow(feat, 1.0/gamma)
    feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        curr_feat = feat[:, curr_channel:curr_channel+num_channels]
        if curr_feat.shape[1] == 1:
            curr_feat = np.repeat(curr_feat, 3, axis=1)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=img_w/PLOT_DPI, h=img_h/PLOT_DPI, forward=False)
        ax = plt.gca()
        for gid in range(len(xy)):
            ellipse = Ellipse(xy=(x[gid], y[gid]), width=scale[gid, 0], height=scale[gid, 1],
                              angle=rot[gid, 0]*180/np.pi, alpha=alpha, ec=None, fc=curr_feat[gid], lw=None)
            ax.add_patch(ellipse)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def visualize_gaussian_position(filepath, images, xy, input_channels, color="#7bf1a8", size=700, every_n=10, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the position of Gaussians using dots.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    xy = xy.detach().cpu().clone().numpy()[::every_n]
    x, y = xy[:, 0] * image_width, xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, f".{save_image_format}", gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(x, y, s=size, c=color, marker='o', alpha=alpha)
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def visualize_added_gaussians(filepath, images, old_xy, new_xy, input_channels, size=500, every_n=5, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the positions of added Gaussians during error-guided progressive optimization.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    old_xy = old_xy.detach().cpu().clone().numpy()[::every_n]
    new_xy = new_xy.detach().cpu().clone().numpy()[::every_n]
    old_x, old_y = old_xy[:, 0] * image_width, old_xy[:, 1] * image_height
    new_x, new_y = new_xy[:, 0] * image_width, new_xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, f".{save_image_format}", gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(old_x, old_y, s=size, c="#ef476f", marker='o', alpha=alpha)  # red
        plt.scatter(new_x, new_y, s=size, c="#06d6a0", marker='o', alpha=alpha)  # green
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def save_error_maps(path, images, gt_images, channels, gamma, save_image_format="jpg"):
    images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/gamma)
    gt_images = torch.pow(gt_images, 1.0/gamma)
    images_sep = separate_image_channels(images, channels)
    gt_images_sep = separate_image_channels(gt_images, channels)
    for idx, (image, gt_image) in enumerate(zip(images_sep, gt_images_sep), 1):
        gt_image, image = gt_image.detach().cpu().clone().numpy(), image.detach().cpu().clone().numpy()
        if gt_image.shape[0] == 1:
            gt_image = np.repeat(gt_image, 3, axis=0)
            image = np.repeat(image, 3, axis=0)
        gt_image, image = gt_image.transpose(1, 2, 0), image.transpose(1, 2, 0)
        suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
        flip_error_map, _, _ = flip_evaluator.evaluate(reference=gt_image, test=image, dynamicRangeString="LDR", inputsRGB=True, applyMagma=True)
        save_image(flip_error_map, f"{path}{suffix}.{save_image_format}")

def render(xy: torch.Tensor, scale: torch.Tensor, rot: torch.Tensor, feat: torch.Tensor, img_size=(512, 512), disable_topk_norm=False, ):

    block_w, block_h = 16, 16
    img_h = img_size[0]
    img_w = img_size[1]

    tile_bounds = ((img_size[1] + block_w - 1) // block_w, (img_size[0] + block_h - 1) // block_h, 1)
    
    tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
    xy_proj, radii, conics, num_tiles_hit = tmp

    tmp = xy_proj, radii, conics, num_tiles_hit, feat, img_h, img_w, block_h, block_w, not disable_topk_norm
    out_image = rasterize_gaussians_sum(*tmp) # type: ignore
    out_image = out_image.view(-1, img_h, img_w, 3).permute(0, 3, 1, 2).contiguous()
    return out_image.squeeze(dim=0)

def render_and_save(xy: torch.Tensor, scale: torch.Tensor, rot: torch.Tensor, feat: torch.Tensor, path, img_size=(512, 512)):
    out_image = render(xy, scale, rot, feat, img_size)
    separate_and_save_images(out_image, [3], path)
    return out_image

