"""
Model builder for skinsa.
Assembles the complete architecture replicating the Sensors 2022 paper:
- Backbone cut at 7x7 feature map (MobileNetV3Large or DenseNet201)
- 16-head Soft-Attention layer (Conv3D layout)
- Max-pooling on plain and attended maps, concatenation, ReLU, Dropout(0.5), Flatten
- Dense(4096) branch over 21 metadata features (age, sex, body site)
- Final Dense(7) softmax output
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model

from skinsa.config import Config, BACKBONE_SPECS
from skinsa.models.attention import SoftAttention
from skinsa.models.backbones import get_cut_backbone


def build_model(
    config: Optional[Config] = None,
    backbone_name: str = "mobilenet_v3_large",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    metadata_dim: int = 21,
    num_classes: int = 7,
    weights: Optional[str] = "imagenet",
    bn_momentum: float = 0.99,
    dropout_rate: float = 0.5,
) -> Model:
    """
    Build the complete Soft-Attention Skin Lesion Classification model.

    Exact Trainable Parameter Targets:
    - MobileNetV3Large: 2,646,679
    - DenseNet201: 17,382,935

    Parameters
    ----------
    config : Config, optional
        Master configuration object. If provided, overrides individual arguments.
    backbone_name : str
        'mobilenet_v3_large' or 'densenet201'.
    input_shape : tuple of int
        Image dimensions (H, W, 3).
    metadata_dim : int
        Number of patient metadata features (default 21).
    num_classes : int
        Number of diagnostic classes (default 7).
    weights : str or None
        Pretrained weights ('imagenet' or None).
    bn_momentum : float
        Batch normalization momentum (0.9 for smoke tests).
    dropout_rate : float
        Dropout rate (default 0.5).

    Returns
    -------
    model : Model
        Multi-input Keras Model: inputs=[image_input, metadata_input], output=prediction.
    """
    if config is not None:
        backbone_name = config.backbone
        input_shape = (config.image_size[0], config.image_size[1], 3)
        metadata_dim = config.metadata_dim
        num_classes = config.num_classes
        weights = config.weights
        bn_momentum = config.bn_momentum
        dropout_rate = config.dropout_rate

    # 1. Inputs
    image_input = layers.Input(shape=input_shape, name="image_input")
    metadata_input = layers.Input(shape=(metadata_dim,), name="metadata_input")

    # 2. Backbone cut at late 7x7 feature map
    cut_backbone, cut_layer, feature_channels = get_cut_backbone(
        backbone_name=backbone_name,
        input_shape=input_shape,
        weights=weights,
        bn_momentum=bn_momentum,
    )
    plain_map = cut_backbone(image_input)  # Shape: (B, 7, 7, C)

    # 3. 16-head Soft-Attention layer (authors' Conv3D layout, no learnable gamma)
    attention_layer = SoftAttention(
        channels=feature_channels,
        num_heads=16,
        kernel_spatial=3,
        name="soft_attention",
    )
    attended_map = attention_layer(plain_map)  # Shape: (B, 7, 7, C)

    # 4. Max-pool plain and attended maps, concatenate, ReLU, Dropout 0.5, Flatten
    # pool_size=(2, 2), strides=(2, 2), padding='same' maps 7x7 -> 4x4
    pool_layer = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="feature_maxpool",
    )
    plain_pooled = pool_layer(plain_map)      # Shape: (B, 4, 4, C)
    attended_pooled = pool_layer(attended_map)  # Shape: (B, 4, 4, C)

    # Concatenate along channel axis: shape (B, 4, 4, 2*C)
    concatenated_visual = layers.Concatenate(
        axis=-1,
        name="concat_plain_attended",
    )([plain_pooled, attended_pooled])

    visual_relu = layers.Activation("relu", name="visual_relu")(concatenated_visual)
    visual_drop = layers.Dropout(dropout_rate, name="visual_dropout")(visual_relu)
    visual_flat = layers.Flatten(name="visual_flatten")(visual_drop)
    # Shape: (B, 4*4*2*C) -> 30,720 for MobileNetV3Large, 4,096 for DenseNet201

    # 5. Metadata branch: Dense(4096) over 21 metadata features
    metadata_dense = layers.Dense(
        4096,
        activation="relu",
        name="metadata_dense_4096",
    )(metadata_input)  # Parameters: 21 * 4096 + 4096 = 90,112

    # 6. Join visual features and metadata branch
    # MobileNet: 30,720 + 4,096 = 34,816
    # DenseNet: 4,096 + 4,096 = 8,192
    fused_features = layers.Concatenate(
        axis=-1,
        name="fusion_visual_metadata",
    )([visual_flat, metadata_dense])

    # 7. Final 7-class softmax output
    # MobileNet params: 34,816 * 7 + 7 = 243,719
    # DenseNet params: 8,192 * 7 + 7 = 57,351
    prediction = layers.Dense(
        num_classes,
        activation="softmax",
        name="diagnostic_softmax_7",
    )(fused_features)

    model = Model(
        inputs=[image_input, metadata_input],
        outputs=prediction,
        name=f"skinsa_{backbone_name}",
    )

    return model


def get_attention_extractor(model: Model) -> Model:
    """
    Build a sub-model that outputs both the prediction and the 2D attention heatmap.
    Used by the UI/UX frontend and plotting module for visual explainability.
    """
    image_input = model.get_layer("image_input").input
    metadata_input = model.get_layer("metadata_input").input
    prediction = model.output

    # Find the plain feature map input to soft attention
    soft_att_layer = model.get_layer("soft_attention")
    plain_map = soft_att_layer.input

    # Compute attention map directly using the trained weights
    # soft_att_layer.call(plain_map, return_attention=True)
    _, attention_map = soft_att_layer.call(plain_map, return_attention=True)

    explainer = Model(
        inputs=[image_input, metadata_input],
        outputs=[prediction, attention_map],
        name="skinsa_explainer",
    )
    return explainer
