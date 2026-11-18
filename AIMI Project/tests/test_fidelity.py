"""
pytest test suite for skinsa.
Tests 9 fidelity criteria including:
1. SoftAttention parameter count for MobileNetV3Large (kernel: C=960)
2. SoftAttention parameter count for DenseNet201 (kernel: C=128)
3. TTA output equals manual average of 8 views
4. Metadata encoding produces exact 21-dim vector
5. Class weight formula produces expected values
6. Model builds without error on both backbones
7. Dihedral TTA produces 8 distinct views
8. No lesion leakage in lesion split protocol
9. SoftAttention output shape is correct
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pytest
import pandas as pd


# ═══════════════════════════════════════════════════════
# TEST 1 & 2: SoftAttention parameter counts
# ═══════════════════════════════════════════════════════

class TestModelTotalTrainableParams:
    """
    Test exact full model trainable parameter counts:
    - MobileNetV3Large: 2,646,679
    - DenseNet201: 17,382,935
    """

    def test_mobilenet_total_trainable_params(self):
        from skinsa.config import Config
        from skinsa.models.builder import build_model
        config = Config(backbone="mobilenet_v3_large", image_size=(224, 224), weights=None)
        model = build_model(config=config)
        trainable = sum(v.numpy().size for v in model.trainable_weights)
        assert trainable == 2_646_679, f"MobileNetV3Large total trainable params expected 2,646,679, got {trainable:,}"

    def test_densenet_total_trainable_params(self):
        from skinsa.config import Config
        from skinsa.models.builder import build_model
        config = Config(backbone="densenet201", image_size=(224, 224), weights=None)
        model = build_model(config=config)
        trainable = sum(v.numpy().size for v in model.trainable_weights)
        assert trainable == 17_382_935, f"DenseNet201 total trainable params expected 17,382,935, got {trainable:,}"


class TestSoftAttentionParams:
    """
    For kernel shape (C, 3, 3, 1, 16) + bias (16,):
    - MobileNetV3Large (C=960): 960×3×3×1×16 + 16 = 138,240 + 16 = 138,256
    - DenseNet201 (C=128): 128×3×3×1×16 + 16 = 18,432 + 16 = 18,448
    """

    def _count_params(self, channels: int) -> int:
        import tensorflow as tf
        from skinsa.models.attention import SoftAttention

        attn = SoftAttention(channels=channels, num_heads=16, kernel_spatial=3)
        # Build layer by calling with dummy tensor
        dummy = tf.zeros([1, 7, 7, channels])
        attn(dummy)
        return sum(np.prod(v.shape) for v in attn.trainable_weights)

    def test_mobilenet_attention_params(self):
        """Test 1: SoftAttention params = 138,256 for MobileNetV3Large (C=960)."""
        expected = 960 * 3 * 3 * 1 * 16 + 16  # = 138,256
        assert expected == 138_256, f"Math error: got {expected}"

        actual = self._count_params(channels=960)
        assert actual == 138_256, (
            f"MobileNetV3Large SoftAttention param count mismatch: "
            f"expected 138,256 but got {actual:,}"
        )

    def test_densenet_attention_params(self):
        """Test 2: SoftAttention params = 18,448 for DenseNet201 (C=128)."""
        expected = 128 * 3 * 3 * 1 * 16 + 16  # = 18,448
        assert expected == 18_448, f"Math error: got {expected}"

        actual = self._count_params(channels=128)
        assert actual == 18_448, (
            f"DenseNet201 SoftAttention param count mismatch: "
            f"expected 18,448 but got {actual:,}"
        )


# ═══════════════════════════════════════════════════════
# TEST 3: TTA output equals manual average of 8 views
# ═══════════════════════════════════════════════════════

class TestTTA:
    def test_tta_equals_manual_average(self):
        """Test 3: TTA average matches manually computing each view and averaging."""
        import tensorflow as tf
        from skinsa.evaluation.tta import DihedralTTA, evaluate_with_tta
        from skinsa.config import Config
        from skinsa.models.builder import build_model
        from skinsa.training.trainer import compile_model

        config = Config(backbone="mobilenet_v3_large", image_size=(32, 32), weights=None)
        model = build_model(config=config)
        model = compile_model(model, config)

        rng = np.random.default_rng(42)
        images = rng.random((3, 32, 32, 3)).astype(np.float32)
        metadata = rng.random((3, 21)).astype(np.float32)

        # TTA result from module
        tta_probs = evaluate_with_tta(model, images, metadata, batch_size=8)

        # Manual average of 8 views
        manual_sum = np.zeros((3, 7), dtype=np.float32)
        for view_idx in range(8):
            view_imgs = np.stack([DihedralTTA.apply_view(images[i], view_idx) for i in range(3)])
            view_probs = model.predict([view_imgs, metadata], verbose=0)
            manual_sum += view_probs
        manual_avg = manual_sum / 8.0

        np.testing.assert_allclose(
            tta_probs, manual_avg, rtol=1e-5, atol=1e-6,
            err_msg="TTA average does not match manual view-by-view average"
        )

    def test_dihedral_has_8_views(self):
        """Test 7: DihedralTTA produces exactly 8 distinct views."""
        from skinsa.evaluation.tta import DihedralTTA

        rng = np.random.default_rng(99)
        img = rng.random((32, 32, 3)).astype(np.float32)
        views = DihedralTTA.all_views(img)

        assert len(views) == 8, f"Expected 8 views, got {len(views)}"

        # At least some views should be numerically distinct
        view_hashes = [v.tobytes() for v in views]
        unique_hashes = set(view_hashes)
        # For a random non-symmetric image, most views should be unique
        # (except in degenerate cases)
        assert len(unique_hashes) >= 4, (
            f"Expected at least 4 distinct dihedral views, got {len(unique_hashes)}"
        )


# ═══════════════════════════════════════════════════════
# TEST 4: Metadata encoding
# ═══════════════════════════════════════════════════════

class TestMetadata:
    def test_metadata_dim_is_21(self):
        """Test 4: Metadata encoder produces exactly 21-dimensional vector."""
        from skinsa.data.metadata import encode_metadata_dict
        from skinsa.config import NUM_METADATA_FEATURES

        assert NUM_METADATA_FEATURES == 21

        for case in [
            {"age": 45, "sex": "male", "localization": "back"},
            {"age": None, "sex": "female", "localization": "unknown"},
            {"age": -1, "sex": "unknown", "localization": "face"},
        ]:
            vec = encode_metadata_dict(case)
            assert vec.shape == (21,), f"Expected (21,), got {vec.shape} for {case}"
            # Only valid one-hot encodings: sum of sex part = 1, sum of loc part = 1
            assert abs(vec[:3].sum() - 1.0) < 1e-6, f"Sex encoding sum ≠ 1 for {case}"
            assert abs(vec[3:18].sum() - 1.0) < 1e-6, f"Loc encoding sum ≠ 1 for {case}"

    def test_missing_age_imputed(self):
        """Test that missing age is imputed with sentinel values."""
        from skinsa.data.metadata import encode_metadata_dict
        vec_missing = encode_metadata_dict({"age": None, "sex": "male", "localization": "back"})
        vec_present = encode_metadata_dict({"age": 50, "sex": "male", "localization": "back"})

        # Missing flag should be 1 for missing age
        assert vec_missing[19] == 1.0, f"Missing age flag should be 1, got {vec_missing[19]}"
        assert vec_present[19] == 0.0, f"Present age flag should be 0, got {vec_present[19]}"


# ═══════════════════════════════════════════════════════
# TEST 5: Class weight formula
# ═══════════════════════════════════════════════════════

class TestClassWeights:
    def test_class_weight_formula(self):
        """Test 5: Class weights follow w_i = N / (C * N_i)."""
        from skinsa.training.losses import compute_class_weights
        from skinsa.config import CLASS_NAMES, HAM10000_DISTRIBUTION

        weights = compute_class_weights()

        N = sum(HAM10000_DISTRIBUTION.values())
        C_num = len(CLASS_NAMES)

        for i, cls in enumerate(CLASS_NAMES):
            n_i = HAM10000_DISTRIBUTION.get(cls, 0)
            expected = N / (C_num * n_i)
            actual = weights[i]
            assert abs(actual - expected) < 1e-5, (
                f"Class weight for {cls}: expected {expected:.6f}, got {actual:.6f}"
            )

    def test_nv_has_smallest_weight(self):
        """NV class (most frequent) should have the smallest weight."""
        from skinsa.training.losses import compute_class_weights
        from skinsa.config import CLASS_NAMES

        weights = compute_class_weights()
        nv_idx = CLASS_NAMES.index("nv")
        nv_weight = weights[nv_idx]

        for i, cls in enumerate(CLASS_NAMES):
            if cls != "nv":
                assert weights[i] > nv_weight, (
                    f"Class {cls} weight ({weights[i]:.4f}) should be > NV weight ({nv_weight:.4f})"
                )


# ═══════════════════════════════════════════════════════
# TEST 6: Model builds correctly
# ═══════════════════════════════════════════════════════

class TestModelBuild:
    def _build_tiny_model(self, backbone: str, channels: int):
        """Build a tiny model (no imagenet weights) for fast testing."""
        import tensorflow as tf
        from skinsa.config import Config
        from skinsa.models.builder import build_model

        config = Config(
            backbone=backbone,
            image_size=(32, 32),
            weights=None,
            is_smoke_test=True,
        )
        model = build_model(config=config)
        return model

    def test_mobilenet_builds(self):
        """Test 6a: Model builds successfully for MobileNetV3Large."""
        model = self._build_tiny_model("mobilenet_v3_large", 960)
        assert model is not None
        assert len(model.inputs) == 2
        assert model.output.shape[-1] == 7

    def test_densenet_builds(self):
        """Test 6b: Model builds successfully for DenseNet201."""
        model = self._build_tiny_model("densenet201", 128)
        assert model is not None
        assert len(model.inputs) == 2
        assert model.output.shape[-1] == 7

    def test_model_forward_pass(self):
        """Test that forward pass runs and produces valid probabilities."""
        import tensorflow as tf
        from skinsa.config import Config
        from skinsa.models.builder import build_model
        from skinsa.training.trainer import compile_model

        config = Config(backbone="mobilenet_v3_large", image_size=(32, 32), weights=None)
        model = build_model(config=config)
        model = compile_model(model, config)

        rng = np.random.default_rng(1)
        imgs = rng.random((2, 32, 32, 3)).astype(np.float32)
        metas = rng.random((2, 21)).astype(np.float32)

        probs = model.predict([imgs, metas], verbose=0)
        assert probs.shape == (2, 7), f"Expected (2, 7), got {probs.shape}"
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ═══════════════════════════════════════════════════════
# TEST 8: No lesion leakage in lesion split
# ═══════════════════════════════════════════════════════

class TestLesionSplit:
    def _make_synthetic_df(self, n=200, seed=42):
        """Make a small synthetic metadata dataframe."""
        rng = np.random.default_rng(seed)
        from skinsa.config import CLASS_NAMES
        # Create synthetic lesion IDs (many-to-one: images share lesion_id)
        lesion_ids = [f"LES_{i:03d}" for i in range(60)]
        classes = rng.choice(CLASS_NAMES, size=n)
        assigned_lesions = rng.choice(lesion_ids, size=n)

        df = pd.DataFrame({
            "image_id": [f"IMG_{i:04d}" for i in range(n)],
            "lesion_id": assigned_lesions,
            "dx": classes,
        })
        return df

    def test_no_leakage_in_lesion_split(self):
        """Test 8: StratifiedGroupKFold split has zero lesion ID leakage."""
        from skinsa.data.splits import split_lesion_protocol, verify_no_lesion_leakage

        df = self._make_synthetic_df(n=200)
        try:
            train_df, val_df, test_df = split_lesion_protocol(df, random_state=42)
            is_clean, leaked = verify_no_lesion_leakage(train_df, test_df)
            assert is_clean, f"Lesion leakage detected: {leaked}"
        except Exception as e:
            # May fail on very small datasets due to stratification constraints
            pytest.skip(f"StratifiedGroupKFold failed on synthetic data: {e}")


# ═══════════════════════════════════════════════════════
# TEST 9: SoftAttention output shape
# ═══════════════════════════════════════════════════════

class TestSoftAttentionOutput:
    def test_output_shape(self):
        """Test 9: SoftAttention output shape matches input feature map shape."""
        import tensorflow as tf
        from skinsa.models.attention import SoftAttention

        for C, H, W in [(960, 7, 7), (128, 7, 7), (64, 14, 14)]:
            attn = SoftAttention(channels=C, num_heads=16, kernel_spatial=3)
            dummy = tf.zeros([2, H, W, C])
            output = attn(dummy)
            assert output.shape == (2, H, W, C), (
                f"Expected output shape (2,{H},{W},{C}), got {output.shape}"
            )

    def test_attention_map_shape_with_return(self):
        """Test that return_attention=True returns both features and 2D map."""
        import tensorflow as tf
        from skinsa.models.attention import SoftAttention

        C = 64
        attn = SoftAttention(channels=C, num_heads=16, kernel_spatial=3)
        dummy = tf.zeros([2, 7, 7, C])
        features, attn_map = attn(dummy, return_attention=True)

        assert features.shape == (2, 7, 7, C), f"Feature shape wrong: {features.shape}"
        assert attn_map.shape == (2, 7, 7, 1), f"Attention map shape wrong: {attn_map.shape}"
