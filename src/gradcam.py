"""
CropGuard AI — Grad-CAM Explainability
=======================================
Implements Grad-CAM (Selvaraju et al., ICCV 2017) for visual explanation
of disease localization decisions.

Target layer: ``model.cbam4.spatial_attention.conv``
The CBAM-4 spatial attention conv is the most interpretable layer:
it explicitly learns *where* in the leaf the disease is located.

Usage
-----
    from src.gradcam import GradCAMWrapper, visualize_gradcam_grid

    cam = GradCAMWrapper(model)
    overlay, heatmap = cam.explain(img_tensor, target_class=1)

References
----------
[1] Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization. ICCV.
[2] Woo et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.utils import CLASS_DISPLAY, IMAGENET_MEAN, IMAGENET_STD, denormalize

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False
    print(
        "⚠️  pytorch-grad-cam not installed. "
        "Run `pip install pytorch-grad-cam` to enable Grad-CAM."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GradCAMWrapper
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMWrapper:
    """
    Convenience wrapper around ``pytorch-grad-cam`` targeting the CBAM-4
    spatial attention convolution layer.

    Args:
        model:       CropGuardCNN instance (must have ``.cbam4``).
        device:      Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        if not _GRADCAM_AVAILABLE:
            raise ImportError("Install pytorch-grad-cam: pip install pytorch-grad-cam")

        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        target_layers = [model.cbam4.spatial_attention.conv]
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def explain(
        self,
        img_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM overlay for one image.

        Args:
            img_tensor:   (C, H, W) normalised image tensor (no batch dim).
            target_class: Class index to explain; uses argmax if None.

        Returns:
            overlay:   (H, W, 3) float32 RGB heatmap blended with original.
            heatmap:   (H, W)    float32 raw CAM values in [0, 1].
        """
        input_batch = img_tensor.unsqueeze(0).to(self.device)

        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            with torch.no_grad():
                pred = self.model(input_batch).argmax(dim=1).item()
            targets = [ClassifierOutputTarget(pred)]

        grayscale_cam = self.cam(input_tensor=input_batch, targets=targets)[0]

        img_rgb = denormalize(img_tensor).astype(np.float32)
        overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        return overlay, grayscale_cam


# ─────────────────────────────────────────────────────────────────────────────
#  Batch Grad-CAM (all 10 classes)
# ─────────────────────────────────────────────────────────────────────────────

def get_one_image_per_class(
    loader: torch.utils.data.DataLoader,
    num_classes: int = 10,
) -> dict:
    """
    Pull one representative image tensor per class from a DataLoader.

    Returns:
        dict mapping class_index → (C, H, W) tensor.
    """
    collected: dict = {}
    for imgs, labels in loader:
        for img, label in zip(imgs, labels):
            cls = label.item()
            if cls not in collected:
                collected[cls] = img
            if len(collected) == num_classes:
                return collected
    return collected


def visualize_gradcam_grid(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str] = CLASS_DISPLAY,
    save_path: str = "./gradcam_all_classes.png",
    device: Optional[torch.device] = None,
) -> None:
    """
    Plot a 3-row grid (Original / Heatmap / Overlay) for all 10 classes.

    Saves the figure to *save_path*.
    """
    cam_wrapper = GradCAMWrapper(model, device)
    class_images = get_one_image_per_class(test_loader, len(class_names))

    n_classes = len(class_names)
    fig, axes = plt.subplots(3, n_classes, figsize=(n_classes * 3, 9))
    fig.suptitle(
        "CropGuard AI — Grad-CAM Disease Localization (ResNet50 + CBAM)",
        fontsize=13, fontweight="bold",
    )

    for cls_idx, disp in enumerate(class_names):
        if cls_idx not in class_images:
            for row in range(3):
                axes[row][cls_idx].axis("off")
            continue

        img_t = class_images[cls_idx]
        overlay, heatmap = cam_wrapper.explain(img_t, target_class=cls_idx)
        img_rgb = denormalize(img_t)

        axes[0][cls_idx].imshow(img_rgb)
        axes[0][cls_idx].set_title(disp, fontsize=7, fontweight="bold")
        axes[0][cls_idx].axis("off")

        axes[1][cls_idx].imshow(heatmap, cmap="jet")
        axes[1][cls_idx].set_title("CAM Heat", fontsize=7)
        axes[1][cls_idx].axis("off")

        axes[2][cls_idx].imshow(overlay)
        axes[2][cls_idx].set_title("Overlay", fontsize=7)
        axes[2][cls_idx].axis("off")

    axes[0][0].set_ylabel("Original",  fontsize=9, labelpad=10)
    axes[1][0].set_ylabel("Grad-CAM",  fontsize=9, labelpad=10)
    axes[2][0].set_ylabel("Overlay",   fontsize=9, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM grid saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-image diagnostic helper (used by Streamlit Diagnose page)
# ─────────────────────────────────────────────────────────────────────────────

def explain_single_image(
    model: nn.Module,
    pil_img: Image.Image,
    predicted_class: int,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM for a PIL image already run through inference.

    Normalises the PIL image internally using ImageNet stats.

    Args:
        model:            CropGuardCNN instance.
        pil_img:          PIL image (RGB, 224×224).
        predicted_class:  Class index from the classifier.
        device:           Inference device.

    Returns:
        (overlay, heatmap) — both (H, W, 3) and (H, W) numpy arrays.
    """
    import torchvision.transforms.functional as TF
    from src.utils import get_val_test_transform

    transform = get_val_test_transform()
    img_tensor = transform(pil_img.resize((224, 224)))

    cam_wrapper = GradCAMWrapper(model, device)
    return cam_wrapper.explain(img_tensor, target_class=predicted_class)
