"""
skinsa configuration module
Defines dataset constants, model architecture targets, training hyper-parameters, and CLI protocols.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Standard 7 classes in HAM10000 (alphabetical order)
CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Detailed clinical descriptions and risk classification
CLASS_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "akiec": {
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "category": "Pre-malignant",
        "risk": "High",
        "color": "#F59E0B",  # Amber
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant",
        "risk": "High",
        "color": "#EF4444",  # Red
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "category": "Benign",
        "risk": "Low",
        "color": "#3B82F6",  # Blue
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "Benign",
        "risk": "Low",
        "color": "#14B8A6",  # Teal
    },
    "mel": {
        "name": "Melanoma",
        "category": "Malignant",
        "risk": "Critical",
        "color": "#DC2626",  # Deep Red
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "category": "Benign",
        "risk": "Low",
        "color": "#10B981",  # Green
    },
    "vasc": {
        "name": "Vascular Lesions",
        "category": "Benign",
        "risk": "Low",
        "color": "#8B5CF6",  # Purple
    },
}

# Real HAM10000 dataset sample distribution (10,015 images total)
HAM10000_DISTRIBUTION: Dict[str, int] = {
    "nv": 6705,     # 66.95%
    "mel": 1113,    # 11.11%
    "bkl": 1099,    # 10.97%
    "bcc": 514,     # 5.13%
    "akiec": 327,   # 3.27%
    "vasc": 142,    # 1.42%
    "df": 115,      # 1.15%
}

TOTAL_HAM10000_SAMPLES: int = 10015

# Metadata categorical vocabulary (exactly produces 21 metadata features)
METADATA_SEX_CATEGORIES: List[str] = ["male", "female", "unknown"]  # 3 features
METADATA_LOCALIZATION_CATEGORIES: List[str] = [
    "abdomen", "acral", "back", "chest", "ear",
    "face", "foot", "genital", "hand", "lower extremity",
    "neck", "scalp", "trunk", "upper extremity", "unknown"
]  # 15 features
# Age features: normalized age (1), age missing indicator (1), age squared / decile (1) -> 3 features
# Total metadata features = 3 (sex) + 15 (localization) + 3 (age) = 21 features
NUM_METADATA_FEATURES: int = 21

# Backbone cut points and architectural counts
BACKBONE_SPECS: Dict[str, Dict] = {
    "mobilenet_v3_large": {
        "display_name": "MobileNetV3Large",
        "keras_app": "MobileNetV3Large",
        "cut_layer": "activation_17",  # Right after expanded_conv_14_expand_bn
        "feature_channels": 960,
        "feature_hw": (7, 7),
        "soft_attention_params": 138256,
        "backbone_trainable_params": 2174592,
        "dense_metadata_params": 90112,
        "dense_head_params": 243719,
        "total_trainable_params": 2646679,
    },
    "densenet201": {
        "display_name": "DenseNet201",
        "keras_app": "DenseNet201",
        "cut_layer": "conv5_block29_1_relu",
        "feature_channels": 128,
        "feature_hw": (7, 7),
        "soft_attention_params": 18448,
        "backbone_trainable_params": 17217024,
        "dense_metadata_params": 90112,
        "dense_head_params": 57351,
        "total_trainable_params": 17382935,
    },
}

# Paper reported baseline metrics (Sensors 2022 Table 4/5)
PAPER_REPORTED_METRICS: Dict[str, Dict[str, float]] = {
    "mobilenet_v3_large": {
        "accuracy": 0.8600,
        "precision": 0.7600,
        "recall": 0.7700,
        "f1": 0.7600,
        "auc": 0.9650,
        "execution_time_ms": 28.5,
    },
    "densenet201": {
        "accuracy": 0.8850,
        "precision": 0.7950,
        "recall": 0.8050,
        "f1": 0.8000,
        "auc": 0.9780,
        "execution_time_ms": 115.0,
    },
}


@dataclass
class Config:
    """Master configuration class for skinsa."""
    # Data & Paths
    data_dir: Path = field(default_factory=lambda: Path("data/ham10000"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    results_bundle_path: Optional[Path] = None

    # Model architecture
    backbone: str = "mobilenet_v3_large"  # 'mobilenet_v3_large' or 'densenet201'
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 7
    attention_heads: int = 16
    metadata_dim: int = NUM_METADATA_FEATURES
    metadata_dense_dim: int = 4096
    dropout_rate: float = 0.5
    weights: Optional[str] = "imagenet"  # 'imagenet' or None
    bn_momentum: float = 0.99  # 0.9 for smoke tests with small batches

    # Training hyper-parameters
    protocol: str = "paper"  # 'paper' (stratified 90/10) or 'lesion' (StratifiedGroupKFold 80/10/10 + TTA)
    batch_size: int = 32
    epochs: int = 250
    lr: float = 1e-3
    epsilon: float = 0.1
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 10
    early_stopping_patience: int = 25
    use_class_weights: bool = True
    random_state: int = 42

    # Smoke testing & synthetic data
    is_smoke_test: bool = False
    synthetic_image_size: Tuple[int, int] = (96, 96)
    synthetic_samples: int = 100

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.results_bundle_path is not None:
            self.results_bundle_path = Path(self.results_bundle_path)

        if self.backbone not in BACKBONE_SPECS:
            raise ValueError(f"Unknown backbone: {self.backbone}. Available: {list(BACKBONE_SPECS.keys())}")

    @property
    def backbone_spec(self) -> Dict:
        return BACKBONE_SPECS[self.backbone]
