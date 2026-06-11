"""
CropGuard AI — Model Architectures
====================================
Contains:
  - ChannelAttention, SpatialAttention, CBAM  (Woo et al., ECCV 2018)
  - CropGuardCNN  : ResNet50 + dual CBAM + classification head
  - CropGuardMobileNet : MobileNetV2 lightweight deployment variant
  - ConvVAE       : Convolutional VAE for Potato-healthy minority-class generation

References
----------
[1] He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
[2] Woo et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
[3] Howard et al. (2017). MobileNets. arXiv:1704.04861.
[4] Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
#  CBAM Attention Modules
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel attention recalibration (CBAM, Woo et al. 2018).

    Uses both AvgPool and MaxPool paths through a shared MLP to compute
    per-channel scaling weights.  Output shape equals input shape.

    Args:
        in_channels:     Number of input feature channels.
        reduction_ratio: Bottleneck reduction factor for the shared MLP.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))  # (B, C)
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))  # (B, C)
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale  # (B, C, H, W)


class SpatialAttention(nn.Module):
    """
    Spatial attention recalibration (CBAM, Woo et al. 2018).

    Produces a (B, 1, H, W) spatial mask that focuses the model on
    disease-relevant regions — the primary Grad-CAM target layer.

    Args:
        kernel_size: Convolution kernel size for the spatial map (default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)       # (B, 1, H, W)
        mx, _ = torch.max(x, dim=1, keepdim=True)      # (B, 1, H, W)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale                               # (B, C, H, W)


class CBAM(nn.Module):
    """
    Full Convolutional Block Attention Module (sequential channel → spatial).

    Drop-in residual block: input and output shapes are identical.

    Args:
        in_channels:     Feature map channels.
        reduction_ratio: Channel attention bottleneck factor.
        spatial_kernel:  Spatial attention convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  CropGuard CNN  (ResNet50 + CBAM)
# ─────────────────────────────────────────────────────────────────────────────

class CropGuardCNN(nn.Module):
    """
    CropGuard AI Core Model: ResNet50 + dual CBAM + 3-layer classifier head.

    Architecture summary
    --------------------
    ResNet50 (ImageNet1K_V1 pretrained)
      └── layer3 (1024 ch, 14×14)  →  CBAM_3
      └── layer4 (2048 ch,  7×7)   →  CBAM_4  ← Grad-CAM target
      └── AdaptiveAvgPool → Flatten → (2048,)
      └── Classifier: Linear(2048→512) → BN → ReLU → Dropout
                      Linear(512→256)  → BN → ReLU → Dropout
                      Linear(256→10)

    Phase 4 HPO-tuned hyperparameters (Trial 23 best config):
      - dropout        = 0.38
      - head LR        = 7.3e-4
      - backbone LR    = 4.2e-6
      - weight decay   = 3.1e-5
      - label smoothing= 0.08

    Args:
        num_classes:      Number of output classes. Default 10.
        dropout:          Dropout probability in classifier head.
        use_cbam:         Whether to attach CBAM modules.
        freeze_backbone:  Freeze all ResNet50 layers (Phase-1 feature extraction).
        head_dims:        Hidden layer sizes for the classification head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.38,
        use_cbam: bool = True,
        freeze_backbone: bool = False,
        head_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if head_dims is None:
            head_dims = [512, 256]
        self.use_cbam = use_cbam

        # ── Backbone ──────────────────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )  # (B, 64, 56, 56)
        self.layer1 = backbone.layer1   # (B, 256,  56, 56)
        self.layer2 = backbone.layer2   # (B, 512,  28, 28)
        self.layer3 = backbone.layer3   # (B, 1024, 14, 14)
        self.layer4 = backbone.layer4   # (B, 2048,  7,  7)

        # ── CBAM (after layer3 and layer4) ────────────────────────────────
        if use_cbam:
            self.cbam3 = CBAM(1024)
            self.cbam4 = CBAM(2048)   # ← primary Grad-CAM target

        # ── Global Average Pool ───────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier Head ───────────────────────────────────────────────
        layers: List[nn.Module] = []
        in_dim = 2048
        for out_dim in head_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all ResNet50 conv layers (Phase-1 feature extraction)."""
        for layer in (self.layer0, self.layer1, self.layer2,
                      self.layer3, self.layer4):
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_top_n(self, n: int = 30) -> None:
        """Unfreeze the last *n* backbone parameter tensors (Phase-2 fine-tuning)."""
        all_params: List[nn.Parameter] = []
        for layer in (self.layer0, self.layer1, self.layer2,
                      self.layer3, self.layer4):
            all_params.extend(layer.parameters())
        for p in all_params[-n:]:
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Unfroze top-{n} backbone params. Trainable: {trainable:,} / {total:,}")

    def freeze_bottom_n(self, n: int = 120) -> None:
        """Freeze the first *n* backbone parameter tensors (Config C differential LR)."""
        all_params: List[nn.Parameter] = []
        for layer in (self.layer0, self.layer1, self.layer2,
                      self.layer3, self.layer4):
            all_params.extend(layer.parameters())
        for i, p in enumerate(all_params):
            p.requires_grad = i >= n

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the CBAM-4 spatial-attention conv layer for Grad-CAM hooks."""
        if self.use_cbam:
            return self.cbam4.spatial_attention.conv
        return self.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
#  CropGuard MobileNet  (lightweight deployment variant)
# ─────────────────────────────────────────────────────────────────────────────

class CropGuardMobileNet(nn.Module):
    """
    Lightweight CropGuard AI variant using MobileNetV2 backbone.

    Designed for Streamlit Cloud deployment (≤512 MB RAM, <10 ms inference).
    ~3.4 M parameters vs ResNet50+CBAM ~25.3 M.

    Args:
        num_classes:     Number of output classes.
        dropout:         Dropout probability.
        freeze_backbone: Freeze MobileNetV2 features initially.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features           # (B, 1280, 7, 7)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        if freeze_backbone:
            self._freeze_backbone()

    # ───────────────────────────────────────────────────────────────
    def _freeze_backbone(self) -> None:
        """Freeze all MobileNetV2 feature layers (Phase-1 feature extraction)."""
        for p in self.features.parameters():
            p.requires_grad = False

    # Public alias so train.py's generic _run_phase works with both model classes
    freeze_backbone = _freeze_backbone

    def unfreeze_top_n(self, n: int = 30) -> None:
        """
        Unfreeze the last n parameter tensors of the backbone for fine-tuning.

        Args:
            n: Number of parameter tensors to unfreeze from the end.
        """
        all_params = list(self.features.parameters())
        for p in all_params[-n:]:
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Unfroze top-{n} backbone params. Trainable: {trainable:,} / {total:,}")

    def freeze_bottom_n(self, n: int = 60) -> None:
        """Freeze the first n backbone parameter tensors (differential LR)."""
        all_params = list(self.features.parameters())
        for i, p in enumerate(all_params):
            p.requires_grad = i >= n

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the last conv layer of MobileNetV2 features for Grad-CAM hooks."""
        # features[-1] is the last ConvBNActivation block; its [0] is the Conv2d
        return self.features[-1][0]

    # ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (B, 1280, 7, 7)
        x = self.gap(x).flatten(1)  # (B, 1280)
        return self.classifier(x)   # (B, num_classes)

# ─────────────────────────────────────────────────────────────────────────────
#  Convolutional VAE  (Potato-healthy minority-class generator)
# ─────────────────────────────────────────────────────────────────────────────

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for synthetic leaf generation.

    Specifically trained on Potato__healthy (152 images) to generate 512+
    synthetic samples, resolving the 12.6× class imbalance.

    Architecture
    ------------
    Encoder: 4× Conv2d(stride=2) + BN + LeakyReLU → Flatten → μ, log_var
    Reparameterization: z = μ + ε · exp(0.5 · log_var),  ε ~ N(0, I)
    Decoder: Linear → Reshape → 4× ConvTranspose2d + BN + ReLU → Tanh

    Training recipe
    ---------------
    - 200 epochs, Adam lr=3e-4, batch=32
    - KL annealing: β linearly 0 → 1 over 50 epochs  (prevents collapse)
    - Achieved FID = 42.7  (target < 50)

    Args:
        latent_dim: Latent space dimensionality. Default 128.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # (B, 256, 4, 4)
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),  # (B, 3, 64, 64)
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + ε · σ,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw *n* samples from the prior z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        imgs = self.decode(z)
        return (imgs + 1.0) / 2.0   # rescale [-1, 1] → [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  VAE loss
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
):
    """
    ELBO = Reconstruction Loss  +  β × KL Divergence.

    Args:
        x_hat:   Reconstructed image  (B, C, H, W).
        x:       Original image       (B, C, H, W).
        mu:      Latent mean           (B, latent_dim).
        log_var: Latent log-variance   (B, latent_dim).
        beta:    KL weight (annealed 0 → 1 over first 50 epochs).

    Returns:
        (total_loss, recon_loss, kl_loss) — all scalars.
    """
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + beta * kl, recon, kl
