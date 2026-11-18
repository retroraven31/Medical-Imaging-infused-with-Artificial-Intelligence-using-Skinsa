"""
Class-weighted loss functions for imbalanced skin lesion classification.
Implements the inverse class-frequency weighting formula from the Sensors 2022 paper:
    w_i = N / (C * N_i)
"""

from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

from skinsa.config import CLASS_NAMES, HAM10000_DISTRIBUTION, TOTAL_HAM10000_SAMPLES


def compute_class_weights(
    class_counts: Optional[Dict[str, int]] = None,
    total_samples: Optional[int] = None,
    class_names: List[str] = CLASS_NAMES,
) -> np.ndarray:
    """
    Calculate class weights using the formula: w_i = N / (C * N_i).

    Parameters
    ----------
    class_counts : dict of {str: int}, optional
        Mapping from class name to frequency. Defaults to HAM10000 distribution.
    total_samples : int, optional
        Total number of samples N. If None, sum of class_counts.
    class_names : list of str
        Ordered list of class names.

    Returns
    -------
    weights : np.ndarray
        Array of shape (C,) with float weights for each class.
    """
    if class_counts is None:
        class_counts = HAM10000_DISTRIBUTION

    c = len(class_names)
    if total_samples is None:
        total_samples = sum(class_counts.values())

    weights = np.zeros(c, dtype=np.float32)
    for idx, name in enumerate(class_names):
        count = class_counts.get(name, 0)
        if count == 0:
            weights[idx] = 1.0
        else:
            weights[idx] = float(total_samples) / float(c * count)

    return weights


@tf.keras.utils.register_keras_serializable(package="skinsa")
class ClassWeightedCrossEntropy(losses.Loss):
    """
    Class-weighted categorical cross-entropy loss.
    Loss for sample j with true class y: L_j = - w_y * log(p_y + eps).
    """

    def __init__(
        self,
        class_weights: Optional[Union[np.ndarray, List[float]]] = None,
        epsilon: float = 1e-7,
        name: str = "class_weighted_cross_entropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if class_weights is None:
            class_weights = compute_class_weights()
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.epsilon = float(epsilon)

    def call(self, y_true, y_pred):
        # Clip probabilities to prevent log(0) NaN
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Ensure y_true is one-hot float
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(self.class_weights)[0])
        else:
            y_true = tf.cast(y_true, tf.float32)

        # Cross-entropy per class: - y_true * log(y_pred)
        ce = - tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # Sample weight = sum_i(y_true_i * w_i)
        sample_weights = tf.reduce_sum(y_true * self.class_weights, axis=-1)

        # Weighted loss
        weighted_loss = ce * sample_weights
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist(),
            "epsilon": self.epsilon,
        })
        return config
