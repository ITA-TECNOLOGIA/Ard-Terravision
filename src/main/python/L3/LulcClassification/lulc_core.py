# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from __future__ import annotations

import math
import numpy as np
import torch
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from PIL import Image
from typing import Optional


CLASS_NAMES = [
    "Urban fabric", "Industrial/Mines", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation", "Agri-Land", "Agro-forestry",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland", "Moors/Heath", "Transitional woodland",
    "Beaches/Sands", "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"
]

COLORS = [
    "#504c4cff", "#ff0000", "#fff5d7", "#f0a0a0", "#e6e64d", "#ffe64d", "#e6cc4d",
    "#f2cca6", "#80ff00", "#00a600", "#4dff00", "#ccf24d", "#a6ff80", "#a6f200",
    "#e6e6e6", "#a6a6ff", "#ccccff", "#00ccf2", "#0080ff"
]

DEFAULT_BAND_NAMES = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]


def overlay_binary_mask(ax, gt_mask, color=(1.0, 0.0, 0.0), alpha=0.35):
    m = (gt_mask == 1)
    if m.sum() == 0:
        return
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = alpha * m.astype(np.float32)
    ax.imshow(overlay)


def create_visualization_pil(
    rgb_data: np.ndarray,
    mask_data: np.ndarray,
    time_idx: int,
    net: str,
    mine_iou: Optional[float] = None,
    mine_pred: Optional[np.ndarray] = None,
    gt_mine: Optional[np.ndarray] = None,
) -> Image.Image:

    # Transparent background
    fig, ax = plt.subplots(
        1, 2,
        figsize=(16, 8),
        facecolor="none",
        gridspec_kw={"wspace": 0.02}  # very small gap between plots
    )

    # Remove outer padding completely
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    rgb = np.clip(rgb_data * 3.5, 0, 1)
    ax[0].imshow(rgb)

    if mine_pred is not None and gt_mine is not None:
        pred = (mine_pred == 1)
        gt = (gt_mine == 1)

        tp = (pred & gt).astype(np.uint8)
        fp = (pred & (~gt)).astype(np.uint8)
        fn = ((~pred) & gt).astype(np.uint8)

        overlay_binary_mask(ax[0], fp, color=(1.0, 0.0, 0.0), alpha=0.35)
        overlay_binary_mask(ax[0], fn, color=(0.0, 1.0, 0.0), alpha=0.35)
        overlay_binary_mask(ax[0], tp, color=(1.0, 1.0, 0.0), alpha=0.45)

        title = f"RGB + TP/FP/FN (t={time_idx})"
    else:
        title = f"RGB (t={time_idx})"

    if mine_iou is not None:
        title += f" | IoU: {mine_iou:.3f}"

    ax[0].set_title(title, fontsize=13, pad=6)
    ax[0].axis("off")

    cmap = ListedColormap(COLORS)
    ax[1].imshow(mask_data, cmap=cmap, vmin=0, vmax=18)
    ax[1].set_title(f"{net} Prediction", fontsize=13, pad=6)
    ax[1].axis("off")

    legend_patches = [
        mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]

    ax[1].legend(
        handles=legend_patches,
        loc="lower left",       # inside axis
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85
    )

    # --- Render tightly & preserve transparency ---
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())  # RGBA
    img = Image.fromarray(buf, mode="RGBA")

    plt.close(fig)

    return img

def build_model(net: str, in_channels: int = 10, classes: int = 19, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if net == "segformer":
        model = smp.Segformer("mit_b4", encoder_weights=None, in_channels=in_channels, classes=classes).to(device)
    elif net == "unet":
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=in_channels, classes=classes).to(device)
    elif net == "deeplabv3plus":
        model = smp.DeepLabV3Plus("efficientnet-b3", encoder_weights=None, in_channels=in_channels, classes=classes).to(device)
    elif net == "unetplusplus":
        model = smp.UnetPlusPlus("resnet50", encoder_weights=None, in_channels=in_channels, classes=classes).to(device)
    else:
        raise ValueError(f"Unknown net='{net}'")

    model.eval()
    return model, device


def load_checkpoint(model, model_path: str, device):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.no_grad()
def infer_multiband_frame(model, device, img_3d: np.ndarray) -> np.ndarray:
    """
    img_3d: (B,H,W) float32 DN scale (0..10000).
    returns: (H,W) uint8 class mask
    """
    if img_3d.ndim != 3:
        raise ValueError(f"Expected (B,H,W), got {img_3d.shape}")

    _, height, width = img_3d.shape

    valid_pixels = img_3d[img_3d > 0]
    if valid_pixels.size > 0 and np.percentile(valid_pixels, 1) > 800:
        img_3d = img_3d - 1000.0
        img_3d = np.clip(img_3d, 0, None)

    target_h = math.ceil(height / 32) * 32
    target_w = math.ceil(width / 32) * 32
    pad_h = target_h - height
    pad_w = target_w - width
    img_padded = np.pad(img_3d, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    inp = torch.from_numpy(img_padded / 10000.0).unsqueeze(0).float().to(device)
    inp = torch.clip(inp, 0, 1)

    output = model(inp)
    pred_padded = torch.argmax(output, dim=1).cpu().numpy().squeeze(0)
    full_mask = pred_padded[:height, :width].astype(np.uint8)
    return full_mask
