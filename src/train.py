"""
CropGuard AI — Training Loop
==============================
Provides:
  - EarlyStopping
  - train_one_epoch  (mixed precision, gradient clipping, MixUp/CutMix)
  - evaluate         (val / test evaluation)
  - get_differential_optimizer (Config C: separate backbone / head LRs)
  - train_model      (Phase-1 + Phase-2 two-stage pipeline)
  - train_vae        (VAE training with KL annealing)
  - generate_synthetic_samples (VAE inference → PNG files)

Usage
-----
    from src.train import train_model
    from src.model import CropGuardCNN
    from src.utils import build_dataloaders, set_seed

    set_seed(42)
    train_loader, val_loader, test_loader = build_dataloaders("./data/raw")
    model = CropGuardCNN().to("cuda")
    history = train_model(model, train_loader, val_loader)
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.model import ConvVAE, vae_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Training Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All training hyper-parameters in one place — no magic numbers in code."""

    # Phase-1 (frozen backbone)
    phase1_epochs: int   = 10
    lr_phase1:     float = 1e-3

    # Phase-2 (fine-tune top-30 backbone layers)
    phase2_epochs:    int   = 20
    lr_phase2:        float = 1e-4
    unfreeze_layers:  int   = 30

    # HPO best config (Optuna Trial 23)
    head_lr:         float = 7.3e-4
    backbone_lr:     float = 4.2e-6
    weight_decay:    float = 3.1e-5
    dropout:         float = 0.38
    label_smoothing: float = 0.08
    batch_size:      int   = 32

    # General
    grad_clip:          float = 1.0
    early_stop_patience: int  = 5
    mixed_precision:    bool  = True
    seed:               int   = 42
    output_dir:         str   = "./checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
#  MixUp / CutMix
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp: linearly interpolate image/label pairs."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_w, cut_h = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: paste a random patch from another image."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = _rand_bbox(x.size(), lam)
    x = x.clone()  # avoid in-place mutation of the original batch
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2))
    return x, y, y[idx], lam


def mixup_criterion(
    criterion, pred, y_a, y_b, lam
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Halt training when validation loss stagnates.

    Saves the best model state internally; call ``restore(model)`` after
    training to reload it.

    Args:
        patience:  Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to reset the counter.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True (stop training) when patience is exhausted.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best-seen weights back into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
#  Optimizer factory
# ─────────────────────────────────────────────────────────────────────────────

def get_differential_optimizer(
    model: nn.Module,
    head_lr: float = 7.3e-4,
    backbone_lr: float = 4.2e-6,
    weight_decay: float = 3.1e-5,
) -> torch.optim.Optimizer:
    """
    Config C (selected): separate learning rates for backbone and head.

    Backbone LR ≪ head LR prevents catastrophic forgetting of low-level
    ImageNet features while allowing the head to converge quickly.

    Importance ranking from HPO study:
      Head LR (0.412) > Dropout (0.287) > Backbone LR (0.143)
    """
    # Support both CropGuardCNN (layer0–layer4) and CropGuardMobileNet (features)
    _backbone_prefixes = ("layer0", "layer1", "layer2", "layer3", "layer4", "features")
    backbone_params = [
        p for n, p in model.named_parameters()
        if any(n.startswith(pfx) for pfx in _backbone_prefixes)
        and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not any(n.startswith(pfx) for pfx in _backbone_prefixes)
        and p.requires_grad
    ]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params,     "lr": head_lr,     "weight_decay": weight_decay},
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Core training / evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_clip: float = 1.0,
    use_mixup: bool = False,
) -> Tuple[float, float, float]:
    """
    One full training epoch with optional mixed precision and MixUp.

    Returns:
        (avg_loss, accuracy, avg_grad_norm)
    """
    model.train()
    total_loss, correct, total, grad_norms = 0.0, 0, 0, []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(gnorm.item())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(gnorm.item() if isinstance(gnorm, torch.Tensor) else gnorm)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total, float(np.mean(grad_norms))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (avg_loss, accuracy, predictions, true_labels, probabilities)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Full two-phase training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Optional[TrainConfig] = None,
    run_name: str = "cropguard",
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Two-phase transfer learning pipeline:
      Phase 1 — frozen backbone, train CBAM + head        (10 epochs, lr=1e-3)
      Phase 2 — unfreeze top-30 layers, fine-tune          (20 epochs, lr=1e-4)

    Args:
        model:        CropGuardCNN or CropGuardMobileNet instance.
        train_loader: Training DataLoader with WeightedRandomSampler.
        val_loader:   Validation DataLoader (no augmentation).
        cfg:          TrainConfig; uses defaults if None.
        run_name:     Prefix for saved checkpoint filenames.
        device:       CUDA/CPU device; auto-detected if None.

    Returns:
        history dict with keys:
          train_loss, val_loss, train_acc, val_acc, lr, grad_norm, phase
    """
    if cfg is None:
        cfg = TrainConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler    = GradScaler() if (cfg.mixed_precision and device.type == "cuda") else None

    history: Dict[str, list] = {
        k: [] for k in
        ["train_loss", "val_loss", "train_acc", "val_acc", "lr", "grad_norm", "phase"]
    }

    def _run_phase(phase_name: str, epochs: int, lr: float, freeze: bool) -> None:
        if freeze:
            model._freeze_backbone()
        else:
            model.unfreeze_top_n(cfg.unfreeze_layers)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        es = EarlyStopping(patience=cfg.early_stop_patience)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, gnorm = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, cfg.grad_clip
            )
            vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["grad_norm"].append(gnorm)
            history["phase"].append(phase_name)

            print(
                f"[{phase_name}] E{epoch:02d}/{epochs} | "
                f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"Acc {tr_acc:.3f}/{vl_acc:.3f} | "
                f"GNorm {gnorm:.3f} | {time.time()-t0:.1f}s"
            )

            if es.step(vl_loss, model):
                print(f"  Early stop at epoch {epoch}.")
                break

        es.restore(model)
        ckpt = {
            "model_state": model.state_dict(),
            "phase": phase_name,
            "config": cfg.__dict__,
        }
        ckpt_path = Path(cfg.output_dir) / f"{run_name}_{phase_name}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"  Checkpoint → {ckpt_path}")

    print("\n=== Phase 1: Feature Extraction (frozen backbone) ===")
    _run_phase("phase1", cfg.phase1_epochs, cfg.lr_phase1, freeze=True)

    print("\n=== Phase 2: Fine-tuning (top-30 layers unfrozen) ===")
    _run_phase("phase2", cfg.phase2_epochs, cfg.lr_phase2, freeze=False)

    return history


# ─────────────────────────────────────────────────────────────────────────────
#  VAE Training
# ─────────────────────────────────────────────────────────────────────────────

def train_vae(
    vae: ConvVAE,
    loader: DataLoader,
    num_epochs: int = 200,
    lr: float = 3e-4,
    kl_anneal_epochs: int = 50,
    save_path: str = "./checkpoints/vae_potato_healthy.pt",
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Train the ConvVAE with KL-divergence annealing.

    KL annealing: β linearly increases 0 → 1 over *kl_anneal_epochs*.
    This prevents posterior collapse (KL → 0) which is a known failure
    mode when training on small datasets (152 images).

    Phase 4 results: FID = 42.7 after 200 epochs (target < 50).

    Args:
        vae:               ConvVAE instance.
        loader:            DataLoader over PotatoHealthyDataset.
        num_epochs:        Total training epochs.
        lr:                Adam learning rate.
        kl_anneal_epochs:  Warm-up length for β annealing.
        save_path:         Where to write the final .pt checkpoint.
        device:            CUDA/CPU.

    Returns:
        history dict with total_loss, recon_loss, kl_loss per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    history: Dict[str, list] = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(1, num_epochs + 1):
        vae.train()
        beta = min(1.0, epoch / kl_anneal_epochs)
        epoch_total, epoch_recon, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0

        for imgs in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = vae(imgs)
            loss, recon, kl    = vae_loss(x_hat, imgs, mu, log_var, beta)
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            n_batches   += 1

        history["total_loss"].append(epoch_total / n_batches)
        history["recon_loss"].append(epoch_recon / n_batches)
        history["kl_loss"].append(epoch_kl    / n_batches)

        if epoch % 20 == 0:
            print(
                f"VAE E{epoch:03d}/{num_epochs} | β={beta:.3f} | "
                f"Total {epoch_total/n_batches:.4f} | "
                f"Recon {epoch_recon/n_batches:.4f} | "
                f"KL {epoch_kl/n_batches:.4f}"
            )

    os.makedirs(Path(save_path).parent, exist_ok=True)
    torch.save(vae.state_dict(), save_path)
    print(f"VAE training complete. Weights saved → {save_path}")
    return history


def generate_synthetic_samples(
    vae: ConvVAE,
    n_samples: int = 512,
    save_dir: str = "./data/augmented/vae_potato_healthy",
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate *n_samples* synthetic Potato__healthy images.

    Samples z ~ N(0, I) and passes through the VAE decoder, then saves
    each image as a PNG. Phase 4 target: 512 samples (achieved FID=42.7).

    Args:
        vae:       Trained ConvVAE (weights loaded).
        n_samples: Number of synthetic images to generate.
        save_dir:  Output directory.
        device:    CUDA/CPU.

    Returns:
        Path to the save directory.
    """
    from torchvision.utils import save_image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    vae.eval().to(device)

    saved, batch_size = 0, 64
    with torch.no_grad():
        while saved < n_samples:
            curr = min(batch_size, n_samples - saved)
            imgs = vae.sample(curr, device)          # [0, 1]
            for i, img in enumerate(imgs):
                save_image(img, f"{save_dir}/synthetic_{saved + i:04d}.png")
            saved += curr

    print(f"Generated {saved} synthetic Potato__healthy images → {save_dir}/")
    return save_dir
