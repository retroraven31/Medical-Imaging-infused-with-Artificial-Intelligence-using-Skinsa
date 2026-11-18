"""
Soft-Attention layer ported to TensorFlow/Keras from the Sensors 2022 paper.
Paper: 'Skin Lesion Classification on Imbalanced Data Using Deep Learning with Soft Attention'

The layer uses the authors' Conv3D layout:
- Kernel shape: (C, 3, 3, 1, 16)
- Stride: (1, 1, C)
- Bias: (16,)
- No learnable gamma parameter
- Parameter count:
  - MobileNetV3Large (C=960): 138,256 parameters
  - DenseNet201 (C=128): 18,448 parameters
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="skinsa")
class SoftAttention(layers.Layer):
    """
    16-head Soft-Attention layer using 3D convolution layout.

    Takes a 2D feature map of shape (batch, H, W, C), reshapes it into
    a 3D volume (batch, C, H, W, 1), and applies a 3D convolution with
    kernel (C, 3, 3, 1, 16) and bias (16,) to produce 16 attention heads.
    The attention heads are normalized and applied to the input feature map
    without any learnable gamma parameter.

    Parameters
    ----------
    channels : int
        Number of input channels C (960 for MobileNetV3Large, 128 for DenseNet201).
    num_heads : int, default=16
        Number of attention heads.
    kernel_spatial : int, default=3
        Spatial kernel size (3x3).
    name : str, optional
        Layer name.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 16,
        kernel_spatial: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.kernel_spatial = int(kernel_spatial)

    def build(self, input_shape):
        c = self.channels
        k = self.kernel_spatial
        h = self.num_heads

        # Kernel shape: (depth=C, height=3, width=3, in_channels=1, out_channels=16)
        # Exactly (C, 3, 3, 1, 16)
        self.kernel = self.add_weight(
            name="conv3d_kernel",
            shape=(c, k, k, 1, h),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="conv3d_bias",
            shape=(h,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention: bool = False):
        """
        Forward pass of Soft-Attention.

        Parameters
        ----------
        inputs : tf.Tensor
            Feature map of shape (B, H, W, C).
        return_attention : bool, default=False
            If True, also returns the 2D attention map (B, H, W, 1).

        Returns
        -------
        tf.Tensor
            Attended feature map of shape (B, H, W, C), or tuple (attended, attention_map).
        """
        x = inputs  # shape (B, H, W, C)
        b = tf.shape(x)[0]
        h_dim = tf.shape(x)[1]
        w_dim = tf.shape(x)[2]

        # Transpose from (B, H, W, C) to (B, C, H, W)
        x_trans = tf.transpose(x, perm=[0, 3, 1, 2])
        # Expand dims to 5D: (B, C, H, W, 1)
        x_5d = tf.expand_dims(x_trans, axis=-1)

        # Pad spatial dimensions H, W with reflection/zeros so 3x3 kernel preserves (H, W)
        pad_size = self.kernel_spatial // 2
        x_padded = tf.pad(
            x_5d,
            paddings=[[0, 0], [0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]],
            mode="CONSTANT",
        )

        # 3D convolution across depth C and spatial 3x3
        # strides: [1, s_depth, s_h, s_w, 1] where s_depth=C covers full depth
        conv = tf.nn.conv3d(
            x_padded,
            self.kernel,
            strides=[1, self.channels, 1, 1, 1],
            padding="VALID",
        )  # Output shape: (B, 1, H, W, 16)
        conv = tf.nn.bias_add(conv, self.bias)

        # Squeeze depth dimension: (B, H, W, 16)
        logits_16 = tf.squeeze(conv, axis=1)

        # Softmax normalization over spatial locations per head
        flat_logits = tf.reshape(logits_16, [b, h_dim * w_dim, self.num_heads])
        spatial_attn = tf.nn.softmax(flat_logits, axis=1)
        attn_heads = tf.reshape(spatial_attn, [b, h_dim, w_dim, self.num_heads])

        # Average over the 16 attention heads to produce scalar spatial attention map
        # Shape: (B, H, W, 1)
        attention_map = tf.reduce_mean(attn_heads, axis=-1, keepdims=True)

        # Scale attention map relative to uniform attention to preserve feature magnitude
        norm_scale = tf.cast(h_dim * w_dim, dtype=x.dtype)
        scaled_attention = attention_map * norm_scale

        # Multiply attention map with plain feature map (no learnable gamma)
        # Shape: (B, H, W, C)
        attended_features = x * scaled_attention

        if return_attention:
            return attended_features, attention_map
        return attended_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "num_heads": self.num_heads,
            "kernel_spatial": self.kernel_spatial,
        })
        return config
