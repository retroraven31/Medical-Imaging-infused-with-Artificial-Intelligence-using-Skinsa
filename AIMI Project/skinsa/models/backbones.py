"""
Backbone loaders and feature extractors for skinsa.
Extracts late 7x7 feature maps from MobileNetV3Large and DenseNet201.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import applications, layers, Model

from skinsa.config import BACKBONE_SPECS


def get_cut_backbone(
    backbone_name: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: Optional[str] = "imagenet",
    bn_momentum: float = 0.99,
) -> Tuple[Model, layers.Layer, int]:
    """
    Load backbone and cut at the specified late 7x7 feature map.

    Parameters
    ----------
    backbone_name : str
        'mobilenet_v3_large' or 'densenet201'.
    input_shape : tuple of int
        Image input shape (H, W, 3).
    weights : str or None
        Pretrained weights ('imagenet' or None).
    bn_momentum : float
        Batch normalization momentum. When smoke testing on CPU with small
        synthetic datasets, setting to 0.9 prevents eval-mode collapse.

    Returns
    -------
    backbone_model : Model
        Truncated model accepting input_shape and outputting 7x7 feature map.
    cut_layer : Layer
        The cut layer object.
    feature_channels : int
        Channel dimension C (960 for MobileNetV3Large, 128 for DenseNet201).
    """
    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Expected one of {list(BACKBONE_SPECS.keys())}")

    spec = BACKBONE_SPECS[backbone_name]
    cut_layer_name = spec["cut_layer"]
    expected_channels = spec["feature_channels"]

    image_input = layers.Input(shape=input_shape, name="image_input")

    if backbone_name == "mobilenet_v3_large":
        base_model = applications.MobileNetV3Large(
            input_tensor=image_input,
            include_top=False,
            weights=weights,
            pooling=None,
        )
    elif backbone_name == "densenet201":
        base_model = applications.DenseNet201(
            input_tensor=image_input,
            include_top=False,
            weights=weights,
            pooling=None,
        )
    else:
        raise ValueError(f"Unsupported backbone {backbone_name}")

    # Adjust BatchNorm momentum if requested (e.g. 0.9 for smoke tests)
    if bn_momentum != 0.99:
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = bn_momentum

    # Find the cut layer
    if backbone_name == "mobilenet_v3_large":
        # Cut at the activation right after expanded_conv_14_expand_bn
        bn_idx = None
        for i, layer in enumerate(base_model.layers):
            if layer.name == "expanded_conv_14_expand_bn" or layer.name.startswith("expanded_conv_14_expand_bn_"):
                bn_idx = i
                break
        if bn_idx is not None and bn_idx + 1 < len(base_model.layers):
            cut_layer = base_model.layers[bn_idx + 1]
        else:
            cut_layer = base_model.get_layer(cut_layer_name)
    else:
        try:
            cut_layer = base_model.get_layer(cut_layer_name)
        except ValueError as e:
            # Fallback search if layer was auto-incremented
            matches = [l for l in base_model.layers if l.name.startswith(cut_layer_name)]
            if matches:
                cut_layer = matches[0]
            else:
                available = [l.name for l in base_model.layers if "conv" in l.name or "relu" in l.name]
                raise ValueError(f"Could not find cut layer '{cut_layer_name}'. Available: {available[:10]}") from e

    cut_model = Model(
        inputs=base_model.inputs,
        outputs=cut_layer.output,
        name=f"{backbone_name}_cut",
    )

    actual_channels = cut_layer.output.shape[-1]
    if actual_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels for cut '{cut_layer_name}', "
            f"but got {actual_channels}."
        )

    return cut_model, cut_layer, expected_channels
