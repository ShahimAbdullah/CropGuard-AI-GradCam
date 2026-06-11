"""
CropGuard AI — Source Package
================================
Import shortcuts:

    from src import CropGuardCNN, CBAM, ConvVAE
    from src import get_treatment, estimate_severity
    from src import CLASS_DISPLAY, TARGET_CLASSES
"""

from src.model import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    CropGuardCNN,
    CropGuardMobileNet,
    ConvVAE,
    vae_loss,
)
from src.utils import (
    TARGET_CLASSES,
    CLASS_DISPLAY,
    CLASS_TO_DISPLAY,
    DISPLAY_TO_CLASS,
    NUM_CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    set_seed,
    get_train_transform,
    get_val_test_transform,
    get_vae_transform,
    denormalize,
    laplacian_variance,
    PlantVillageDataset,
    PotatoHealthyDataset,
    build_weighted_sampler,
    build_dataloaders,
)
from src.train import (
    TrainConfig,
    EarlyStopping,
    train_one_epoch,
    evaluate,
    get_differential_optimizer,
    train_model,
    train_vae,
    generate_synthetic_samples,
)
from src.severity import estimate_severity, batch_severity, severity_to_colour
from src.treatment import get_treatment, TREATMENT_DB, list_diseases

__all__ = [
    # model
    "ChannelAttention", "SpatialAttention", "CBAM",
    "CropGuardCNN", "CropGuardMobileNet", "ConvVAE", "vae_loss",
    # utils
    "TARGET_CLASSES", "CLASS_DISPLAY", "CLASS_TO_DISPLAY", "DISPLAY_TO_CLASS",
    "NUM_CLASSES", "IMG_SIZE", "BATCH_SIZE", "IMAGENET_MEAN", "IMAGENET_STD",
    "set_seed", "get_train_transform", "get_val_test_transform", "get_vae_transform",
    "denormalize", "laplacian_variance",
    "PlantVillageDataset", "PotatoHealthyDataset",
    "build_weighted_sampler", "build_dataloaders",
    # train
    "TrainConfig", "EarlyStopping",
    "train_one_epoch", "evaluate",
    "get_differential_optimizer", "train_model",
    "train_vae", "generate_synthetic_samples",
    # severity
    "estimate_severity", "batch_severity", "severity_to_colour",
    # treatment
    "get_treatment", "TREATMENT_DB", "list_diseases",
]
