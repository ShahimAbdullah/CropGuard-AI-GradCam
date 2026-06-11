"""
CropGuard AI — Unit Tests
==========================
Tests for model architecture, utilities, severity, and treatment modules.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    ChannelAttention, SpatialAttention, CBAM,
    CropGuardCNN, CropGuardMobileNet, ConvVAE, vae_loss,
)
from src.utils import (
    TARGET_CLASSES, CLASS_DISPLAY, NUM_CLASSES,
    set_seed, get_train_transform, get_val_test_transform,
    denormalize, IMAGENET_MEAN, IMAGENET_STD,
)
from src.severity import estimate_severity, severity_to_colour
from src.treatment import get_treatment, TREATMENT_DB, list_diseases


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def dummy_batch(device):
    set_seed(42)
    return torch.randn(4, 3, 224, 224).to(device)


@pytest.fixture(scope="module")
def dummy_batch_small(device):
    """64×64 batch for VAE tests."""
    return torch.randn(4, 3, 64, 64).to(device)


@pytest.fixture(scope="module")
def cropguard_model(device):
    set_seed(42)
    model = CropGuardCNN(
        num_classes=NUM_CLASSES,
        dropout=0.38,
        use_cbam=True,
        freeze_backbone=False,
        head_dims=[512, 256],
    ).to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  CBAM Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCBAM:
    def test_channel_attention_shape_1024(self, device):
        ca = ChannelAttention(1024).to(device)
        x  = torch.randn(2, 1024, 14, 14).to(device)
        y  = ca(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_channel_attention_shape_2048(self, device):
        ca = ChannelAttention(2048).to(device)
        x  = torch.randn(2, 2048, 7, 7).to(device)
        y  = ca(x)
        assert y.shape == x.shape

    def test_spatial_attention_shape(self, device):
        sa = SpatialAttention().to(device)
        x  = torch.randn(2, 2048, 7, 7).to(device)
        y  = sa(x)
        assert y.shape == x.shape

    def test_cbam_shape_preserving(self, device):
        cbam = CBAM(2048).to(device)
        x    = torch.randn(2, 2048, 7, 7).to(device)
        y    = cbam(x)
        assert y.shape == x.shape

    def test_cbam_output_range(self, device):
        cbam = CBAM(256).to(device)
        x    = torch.randn(2, 256, 14, 14).to(device)
        y    = cbam(x)
        # Attention scaling should not explode values
        assert not torch.isnan(y).any(), "NaN in CBAM output"
        assert not torch.isinf(y).any(), "Inf in CBAM output"

    def test_cbam_param_count(self, device):
        cbam = CBAM(2048, reduction_ratio=16)
        params = sum(p.numel() for p in cbam.parameters())
        # Should be roughly (2048/16)*2*2048 + 7*7*2 ≈ 262k params
        assert params > 0
        assert params < 1_000_000, f"CBAM has unexpectedly many params: {params}"


# ─────────────────────────────────────────────────────────────────────────────
#  CropGuardCNN Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCropGuardCNN:
    def test_output_shape(self, cropguard_model, dummy_batch):
        with torch.no_grad():
            out = cropguard_model(dummy_batch)
        assert out.shape == (4, NUM_CLASSES), f"Got {out.shape}"

    def test_no_nan_output(self, cropguard_model, dummy_batch):
        with torch.no_grad():
            out = cropguard_model(dummy_batch)
        assert not torch.isnan(out).any()

    def test_freeze_backbone(self, device):
        model = CropGuardCNN(
            num_classes=NUM_CLASSES, use_cbam=True,
            freeze_backbone=True,
        ).to(device)
        backbone_grads = [
            p.requires_grad
            for n, p in model.named_parameters()
            if any(l in n for l in ("layer0","layer1","layer2","layer3","layer4"))
        ]
        assert not any(backbone_grads), "Backbone should be frozen"

    def test_unfreeze_top_n(self, device):
        model = CropGuardCNN(
            num_classes=NUM_CLASSES, use_cbam=True,
            freeze_backbone=True,
        ).to(device)
        model.unfreeze_top_n(30)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_freeze_bottom_n(self, device):
        model = CropGuardCNN(num_classes=NUM_CLASSES, use_cbam=True).to(device)
        model.freeze_bottom_n(120)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        assert 0 < trainable < total

    def test_gradcam_target_layer(self, cropguard_model):
        layer = cropguard_model.get_gradcam_target_layer()
        assert isinstance(layer, nn.Module)

    def test_single_image_inference(self, cropguard_model, device):
        x = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = cropguard_model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_forward_backward(self, device):
        set_seed(0)
        model = CropGuardCNN(
            num_classes=NUM_CLASSES, use_cbam=True, freeze_backbone=True
        ).to(device)
        model.train()
        x      = torch.randn(2, 3, 224, 224).to(device)
        labels = torch.randint(0, NUM_CLASSES, (2,)).to(device)
        loss   = nn.CrossEntropyLoss()(model(x), labels)
        loss.backward()
        assert loss.item() > 0

    def test_no_cbam_mode(self, device):
        model = CropGuardCNN(num_classes=NUM_CLASSES, use_cbam=False).to(device)
        x     = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_total_params(self, cropguard_model):
        total = sum(p.numel() for p in cropguard_model.parameters())
        # ResNet50+CBAM+head ≈ 25–26 M
        assert 23_000_000 < total < 27_000_000, f"Unexpected param count: {total:,}"


# ─────────────────────────────────────────────────────────────────────────────
#  CropGuardMobileNet Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCropGuardMobileNet:
    def test_output_shape(self, device):
        model = CropGuardMobileNet(num_classes=NUM_CLASSES).to(device)
        model.eval()
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_param_count_smaller_than_resnet(self, device):
        mobile  = CropGuardMobileNet(num_classes=NUM_CLASSES).to(device)
        resnet  = CropGuardCNN(num_classes=NUM_CLASSES).to(device)
        mobile_p = sum(p.numel() for p in mobile.parameters())
        resnet_p = sum(p.numel() for p in resnet.parameters())
        assert mobile_p < resnet_p, "MobileNet should have fewer params than ResNet50"


# ─────────────────────────────────────────────────────────────────────────────
#  ConvVAE Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConvVAE:
    def test_forward_shapes(self, device, dummy_batch_small):
        vae = ConvVAE(latent_dim=128).to(device)
        x_hat, mu, log_var = vae(dummy_batch_small)
        assert x_hat.shape == dummy_batch_small.shape
        assert mu.shape == (4, 128)
        assert log_var.shape == (4, 128)

    def test_elbo_loss_positive(self, device, dummy_batch_small):
        vae = ConvVAE(latent_dim=128).to(device)
        x_hat, mu, log_var = vae(dummy_batch_small)
        loss, recon, kl = vae_loss(x_hat, dummy_batch_small, mu, log_var, beta=1.0)
        assert loss.item() > 0
        assert recon.item() > 0
        assert kl.item() > 0

    def test_kl_annealing_zero(self, device, dummy_batch_small):
        vae = ConvVAE(latent_dim=128).to(device)
        x_hat, mu, log_var = vae(dummy_batch_small)
        loss_0, recon_0, _ = vae_loss(x_hat, dummy_batch_small, mu, log_var, beta=0.0)
        assert abs(loss_0.item() - recon_0.item()) < 1e-4

    def test_sample(self, device):
        vae    = ConvVAE(latent_dim=128).to(device)
        vae.eval()
        imgs   = vae.sample(8, device)
        assert imgs.shape == (8, 3, 64, 64)
        assert imgs.min() >= 0.0
        assert imgs.max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Utils Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_class_constants(self):
        assert len(TARGET_CLASSES) == 10
        assert len(CLASS_DISPLAY)  == 10
        assert NUM_CLASSES         == 10

    def test_train_transform_output_shape(self):
        from PIL import Image
        tfm = get_train_transform()
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        t   = tfm(img)
        assert t.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from PIL import Image
        tfm = get_val_test_transform()
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        t   = tfm(img)
        assert t.shape == (3, 224, 224)

    def test_denormalize_range(self):
        t = torch.randn(3, 224, 224)
        out = denormalize(t)
        assert out.shape == (224, 224, 3)
        assert 0.0 <= out.min()
        assert out.max() <= 1.0

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# ─────────────────────────────────────────────────────────────────────────────
#  Severity Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverity:
    def test_severity_colour_all_stages(self):
        for stage in ("Early", "Moderate", "Severe", "Critical", "Unknown"):
            colour = severity_to_colour(stage)
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_estimate_severity_missing_file(self):
        result = estimate_severity("/nonexistent/path/leaf.jpg")
        assert result["severity_pct"] == 0.0
        assert result["stage"] == "Unknown"

    def test_estimate_severity_synthetic_green_image(self, tmp_path):
        import cv2
        img_path = str(tmp_path / "green_leaf.jpg")
        # Create a solid green image (healthy leaf proxy)
        green = np.zeros((224, 224, 3), dtype=np.uint8)
        green[:, :] = (0, 180, 0)  # BGR green
        cv2.imwrite(img_path, green)
        result = estimate_severity(img_path)
        assert "severity_pct" in result
        assert "stage" in result
        assert result["severity_pct"] >= 0.0
        assert result["stage"] in ("Early", "Moderate", "Severe", "Critical", "Unknown")


# ─────────────────────────────────────────────────────────────────────────────
#  Treatment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTreatment:
    def test_all_classes_in_db(self):
        for cls in TARGET_CLASSES:
            assert cls in TREATMENT_DB, f"Missing: {cls}"

    def test_all_db_entries_have_required_keys(self):
        required = {"common_name", "pathogen", "cause",
                    "symptoms", "fungicide", "organic", "prevention"}
        for cls, info in TREATMENT_DB.items():
            missing = required - set(info.keys())
            assert not missing, f"Class {cls} missing keys: {missing}"

    def test_get_treatment_returns_string(self):
        report = get_treatment("Tomato___Early_blight", severity_stage="Moderate")
        assert isinstance(report, str)
        assert len(report) > 100

    def test_get_treatment_as_dict(self):
        result = get_treatment("Tomato___Late_blight", severity_stage="Severe", as_dict=True)
        assert isinstance(result, dict)
        assert "urgency" in result

    def test_get_treatment_unknown_class(self):
        result = get_treatment("Unknown___class", severity_stage="Early")
        assert isinstance(result, str)

    def test_list_diseases(self):
        diseases = list_diseases()
        assert len(diseases) == 10
        assert "Tomato___Early_blight" in diseases

    @pytest.mark.parametrize("stage", ["Early", "Moderate", "Severe", "Critical"])
    def test_all_severity_stages(self, stage):
        report = get_treatment("Potato___Late_blight", severity_stage=stage)
        assert isinstance(report, str)
        assert stage.upper() in report or stage in report or len(report) > 50
