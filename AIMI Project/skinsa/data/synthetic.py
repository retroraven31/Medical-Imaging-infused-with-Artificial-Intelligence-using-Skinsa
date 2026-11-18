"""
Synthetic data generator for CPU smoke testing.
Generates small banner-marked synthetic dermoscopy images (default 96 px)
with explicit 'SYNTHETIC DATA' watermark to ensure synthetic metrics are
never conflated with real experimental results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from skinsa.config import (
    CLASS_NAMES,
    HAM10000_DISTRIBUTION,
    METADATA_LOCALIZATION_CATEGORIES,
    METADATA_SEX_CATEGORIES,
)
from skinsa.data.metadata import encode_metadata_dict


# Synthetic class color signatures for visual distinction
CLASS_SYNTHETIC_HUES: Dict[str, Tuple[int, int, int]] = {
    "akiec": (210, 140, 120),  # Scaly pinkish
    "bcc": (190, 80, 80),      # Pearly reddish
    "bkl": (130, 100, 70),     # Brownish crust
    "df": (160, 120, 100),     # Firm dome tan
    "mel": (40, 30, 30),       # Asymmetric dark black/brown
    "nv": (140, 90, 60),       # Uniform brown nevus
    "vasc": (180, 20, 50),     # Deep vascular red
}


def generate_synthetic_image(
    class_name: str,
    size: Tuple[int, int] = (96, 96),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a single banner-marked synthetic dermoscopy image at requested size.

    Parameters
    ----------
    class_name : str
        One of the 7 HAM10000 diagnosis classes.
    size : tuple of int
        Image dimensions (H, W). Default (96, 96).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    img_array : np.ndarray of shape (H, W, 3) in range [0, 255] uint8.
    """
    rng = np.random.default_rng(seed)
    h, w = size

    # Base skin tone background (warm beige/peach)
    base_r = int(rng.integers(210, 240))
    base_g = int(rng.integers(180, 210))
    base_b = int(rng.integers(160, 190))

    img = Image.new("RGB", (w, h), color=(base_r, base_g, base_b))
    draw = ImageDraw.Draw(img)

    # Draw synthetic lesion oval in center
    lesion_color = CLASS_SYNTHETIC_HUES.get(class_name, (100, 70, 50))
    center_x = w // 2 + int(rng.integers(-4, 5))
    center_y = h // 2 + int(rng.integers(-4, 5))
    radius_x = int(w * rng.uniform(0.20, 0.38))
    radius_y = int(h * rng.uniform(0.20, 0.38))

    bbox = [
        center_x - radius_x,
        center_y - radius_y,
        center_x + radius_x,
        center_y + radius_y,
    ]
    draw.ellipse(bbox, fill=lesion_color, outline=(lesion_color[0]//2, lesion_color[1]//2, lesion_color[2]//2))

    # Add dermoscopy peripheral vignette
    draw.rectangle([0, 0, w, 6], fill=(40, 40, 40))
    draw.rectangle([0, h - 6, w, h], fill=(40, 40, 40))

    # Explicit Banner Watermark
    banner_color = (220, 38, 38)  # Bright red
    draw.rectangle([0, h - 14, w, h], fill=banner_color)
    draw.text((2, h - 13), f"SYNTHETIC:{class_name}", fill=(255, 255, 255))

    return np.array(img, dtype=np.uint8)


def create_synthetic_dataset(
    num_samples: int = 100,
    size: Tuple[int, int] = (96, 96),
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate an entire synthetic dataset matching HAM10000 schema and metadata.

    Returns
    -------
    images : np.ndarray of shape (N, H, W, 3) normalized to [0, 1] float32
    metadata : np.ndarray of shape (N, 21) float32
    labels : np.ndarray of shape (N, 7) one-hot float32
    df : pd.DataFrame containing image_id, lesion_id, dx, age, sex, localization
    """
    rng = np.random.default_rng(seed)

    # Class probability distribution matching HAM10000
    classes = CLASS_NAMES
    probs = [HAM10000_DISTRIBUTION[c] / sum(HAM10000_DISTRIBUTION.values()) for c in classes]

    sampled_classes = rng.choice(classes, size=num_samples, p=probs)

    records = []
    images = np.zeros((num_samples, size[0], size[1], 3), dtype=np.float32)
    metadata_vecs = np.zeros((num_samples, 21), dtype=np.float32)
    one_hot_labels = np.zeros((num_samples, len(classes)), dtype=np.float32)

    # Multiple images per lesion to test lesion_id grouping
    unique_lesions = [f"SYNTH_LES_{i:04d}" for i in range(max(10, num_samples // 2))]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        cls = sampled_classes[i]
        cls_idx = classes.index(cls)
        one_hot_labels[i, cls_idx] = 1.0

        img_uint8 = generate_synthetic_image(cls, size=size, seed=seed + i)
        images[i] = img_uint8.astype(np.float32) / 255.0

        image_id = f"SYNTH_IMG_{i:05d}"
        lesion_id = rng.choice(unique_lesions)
        age = int(rng.integers(15, 85)) if rng.random() > 0.05 else None
        sex = rng.choice(METADATA_SEX_CATEGORIES[:2]) if rng.random() > 0.05 else "unknown"
        loc = rng.choice(METADATA_LOCALIZATION_CATEGORIES[:12]) if rng.random() > 0.05 else "unknown"

        meta_dict = {"age": age, "sex": sex, "localization": loc}
        metadata_vecs[i] = encode_metadata_dict(meta_dict)

        records.append({
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": cls,
            "dx_type": "consensus",
            "age": age,
            "sex": sex,
            "localization": loc,
            "dataset": "synthetic_smoke",
        })

        if output_dir is not None:
            Image.fromarray(img_uint8).save(output_dir / f"{image_id}.jpg")

    df = pd.DataFrame(records)
    if output_dir is not None:
        df.to_csv(output_dir / "HAM10000_metadata.csv", index=False)

    return images, metadata_vecs, one_hot_labels, df
