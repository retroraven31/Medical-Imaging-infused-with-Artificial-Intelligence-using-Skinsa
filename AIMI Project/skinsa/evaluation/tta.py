"""
8-view Dihedral Test-Time Augmentation (TTA).
Dihedral group D4 has 8 elements (identity + 3 rotations, each × horizontal flip).
TTA averages softmax outputs over all 8 views, as applied in the enhanced protocol.
"""

from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf


class DihedralTTA:
    """
    8-view dihedral test-time augmentation.
    The 8 views are: {id, rot90, rot180, rot270} × {original, flipped_LR}.
    """

    @staticmethod
    def apply_view(image: np.ndarray, view_idx: int) -> np.ndarray:
        """
        Apply one of the 8 dihedral transformations to a single image.

        Parameters
        ----------
        image : np.ndarray of shape (H, W, 3) in range [0, 1].
        view_idx : int in [0, 7].

        Returns
        -------
        transformed : np.ndarray of shape (H, W, 3).
        """
        flipped = view_idx >= 4
        rotation_k = view_idx % 4

        img = image.copy()
        if flipped:
            img = img[:, ::-1, :]  # Horizontal flip
        if rotation_k > 0:
            img = np.rot90(img, k=rotation_k, axes=(0, 1))
        return img

    @staticmethod
    def undo_view(prediction: np.ndarray, view_idx: int) -> np.ndarray:
        """
        Undo the spatial transformation on the attention heatmap.
        Predictions (class probabilities) are rotation-invariant by nature.
        This function is included for completeness with heatmap inversion.
        Returns the prediction unchanged.
        """
        return prediction

    @classmethod
    def all_views(cls, image: np.ndarray) -> List[np.ndarray]:
        """
        Return all 8 dihedral views of a single image.

        Returns
        -------
        views : list of 8 np.ndarray each of shape (H, W, 3)
        """
        return [cls.apply_view(image, i) for i in range(8)]

    @classmethod
    def batch_views(cls, images: np.ndarray) -> np.ndarray:
        """
        Produce all 8 TTA views for a batch of images.

        Parameters
        ----------
        images : np.ndarray of shape (N, H, W, 3).

        Returns
        -------
        views : np.ndarray of shape (N, 8, H, W, 3).
        """
        n, h, w, c = images.shape
        views = np.zeros((n, 8, h, w, c), dtype=images.dtype)
        for view_idx in range(8):
            for i in range(n):
                views[i, view_idx] = cls.apply_view(images[i], view_idx)
        return views


def evaluate_with_tta(
    model: tf.keras.Model,
    images: np.ndarray,
    metadata: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Evaluate a model on images using 8-view dihedral TTA.
    Averages softmax outputs over all 8 views.

    Parameters
    ----------
    model : keras.Model
        Multi-input model expecting [images, metadata] inputs.
    images : np.ndarray of shape (N, H, W, 3)
    metadata : np.ndarray of shape (N, 21)
    batch_size : int

    Returns
    -------
    avg_probs : np.ndarray of shape (N, num_classes)
        Averaged softmax probabilities across all 8 views.
    """
    n = len(images)
    num_classes = model.output.shape[-1]
    all_view_probs = np.zeros((n, 8, num_classes), dtype=np.float32)

    for view_idx in range(8):
        # Apply transformation to all images in this view
        view_images = np.zeros_like(images)
        for i in range(n):
            view_images[i] = DihedralTTA.apply_view(images[i], view_idx)

        # Predict in mini-batches to avoid OOM
        probs_view = np.zeros((n, num_classes), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = view_images[start:end]
            batch_meta = metadata[start:end]
            batch_probs = model.predict(
                [batch_imgs, batch_meta],
                verbose=0,
                batch_size=batch_size,
            )
            probs_view[start:end] = batch_probs

        all_view_probs[:, view_idx, :] = probs_view

    # Average over 8 views
    avg_probs = all_view_probs.mean(axis=1)
    return avg_probs


def evaluate_without_tta(
    model: tf.keras.Model,
    images: np.ndarray,
    metadata: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Baseline evaluation without TTA.
    Returns softmax probabilities for each image.
    """
    n = len(images)
    num_classes = model.output.shape[-1]
    probs = np.zeros((n, num_classes), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_probs = model.predict(
            [images[start:end], metadata[start:end]],
            verbose=0,
            batch_size=batch_size,
        )
        probs[start:end] = batch_probs

    return probs
