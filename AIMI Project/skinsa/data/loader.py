"""
HAM10000 data loading module.
Provides utilities for loading images, metadata, and creating TensorFlow datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from skinsa.config import CLASS_NAMES, NUM_METADATA_FEATURES
from skinsa.data.metadata import encode_metadata_dataframe


def load_ham10000_metadata(
    data_dir: Path,
) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV from data_dir.
    Expects a 'HAM10000_metadata.csv' file in data_dir.

    Returns
    -------
    df : pd.DataFrame with 10,015 rows and columns:
        image_id, lesion_id, dx, dx_type, age, sex, localization, dataset
    """
    csv_path = data_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata CSV not found at {csv_path}. "
            "Please download HAM10000 from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 "
            "and place HAM10000_metadata.csv in the data directory."
        )
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Validate expected columns
    required_cols = ["image_id", "lesion_id", "dx"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_image_to_array(
    image_path: Path,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load a single image from disk and resize to target_size.
    Returns float32 array in range [0, 1] with shape (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def build_tf_dataset(
    df: pd.DataFrame,
    image_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    is_training: bool = True,
) -> tf.data.Dataset:
    """
    Build a TensorFlow Dataset from a metadata DataFrame and image directory.
    Returns batches of ((images, metadata), one_hot_labels).

    Parameters
    ----------
    df : pd.DataFrame
        Metadata dataframe with image_id, dx, age, sex, localization columns.
    image_dir : Path
        Directory containing dermoscopy JPEG images.
    target_size : tuple of int
        Image resize target (H, W).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the dataset.
    seed : int
        Shuffle seed.
    is_training : bool
        When True, applies basic augmentation (flips, slight rotations).

    Returns
    -------
    dataset : tf.data.Dataset
        Yields (images, metadata), labels where:
        - images: float32 (B, H, W, 3) in [0, 1]
        - metadata: float32 (B, 21)
        - labels: float32 (B, 7) one-hot
    """
    # Encode metadata
    meta_matrix = encode_metadata_dataframe(df)

    # Build label one-hot matrix
    n = len(df)
    labels = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
    for idx, cls in enumerate(df["dx"].values):
        if cls in CLASS_NAMES:
            labels[idx, CLASS_NAMES.index(cls)] = 1.0

    # Collect image paths
    image_paths = []
    for image_id in df["image_id"].values:
        # Images may be in subdirectories: HAM10000_images_part1 and HAM10000_images_part2
        candidates = [
            image_dir / f"{image_id}.jpg",
            image_dir / "HAM10000_images_part1" / f"{image_id}.jpg",
            image_dir / "HAM10000_images_part2" / f"{image_id}.jpg",
            image_dir / f"{image_id}.JPG",
        ]
        found = False
        for candidate in candidates:
            if candidate.exists():
                image_paths.append(str(candidate))
                found = True
                break
        if not found:
            # Fallback: empty string signals missing image
            image_paths.append("")

    image_paths = np.array(image_paths)
    h, w = target_size

    def load_and_preprocess(img_path_bytes, meta_vec, label_vec):
        img_path = img_path_bytes.numpy().decode("utf-8")
        if img_path:
            img_data = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img_data, channels=3)
            img = tf.image.resize(img, [h, w])
            img = tf.cast(img, tf.float32) / 255.0
        else:
            img = tf.zeros([h, w, 3], dtype=tf.float32)
        return img, meta_vec, label_vec

    def augment(img, meta, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, meta, label

    def wrap_load(path, meta, label):
        img, meta, label = tf.py_function(
            func=load_and_preprocess,
            inp=[path, meta, label],
            Tout=[tf.float32, tf.float32, tf.float32],
        )
        img.set_shape([h, w, 3])
        meta.set_shape([NUM_METADATA_FEATURES])
        label.set_shape([len(CLASS_NAMES)])
        return img, meta, label

    dataset = tf.data.Dataset.from_tensor_slices((
        image_paths,
        meta_matrix.astype(np.float32),
        labels,
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(n, 5000), seed=seed)

    dataset = dataset.map(wrap_load, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Restructure output for multi-input model: ((images, metadata), labels)
    dataset = dataset.map(lambda img, meta, label: ((img, meta), label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_tf_dataset_from_arrays(
    images: np.ndarray,
    metadata: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    is_training: bool = False,
) -> tf.data.Dataset:
    """
    Build a TensorFlow Dataset directly from pre-loaded numpy arrays.
    Used for synthetic smoke tests.
    """
    n = len(images)
    dataset = tf.data.Dataset.from_tensor_slices((
        images.astype(np.float32),
        metadata.astype(np.float32),
        labels.astype(np.float32),
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n, seed=seed)

    dataset = dataset.map(lambda img, meta, label: ((img, meta), label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
