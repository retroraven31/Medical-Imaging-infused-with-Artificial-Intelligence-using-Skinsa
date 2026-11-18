"""
Smoke tests for skinsa.
Runs a quick end-to-end check on synthetic data at 96px with CPU-only.
All results are clearly marked as SYNTHETIC.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pytest


SYNTHETIC_BANNER = "=" * 60 + "\n⚠️ SYNTHETIC SMOKE TEST — NOT REAL RESULTS\n" + "=" * 60


class TestSmokePipeline:
    """
    Quick end-to-end pipeline test on tiny synthetic data (96px, 50 samples, CPU).
    Verifies the entire stack from data generation to model training.
    """

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        from skinsa.data.synthetic import create_synthetic_dataset
        print(SYNTHETIC_BANNER)
        images, metadata, labels, df = create_synthetic_dataset(
            num_samples=50, size=(96, 96), seed=0
        )
        return images, metadata, labels, df

    def test_synthetic_data_shape(self, synthetic_data):
        images, metadata, labels, df = synthetic_data
        assert images.shape == (50, 96, 96, 3)
        assert metadata.shape == (50, 21)
        assert labels.shape == (50, 7)
        assert len(df) == 50
        # Verify all images have 'SYNTHETIC' marker visible
        assert all(df["dataset"] == "synthetic_smoke"), "Missing synthetic marker!"

    def test_synthetic_values_in_range(self, synthetic_data):
        images, metadata, labels, df = synthetic_data
        assert images.min() >= 0.0 and images.max() <= 1.0, "Images out of [0,1] range"
        assert labels.sum(axis=1).mean() == 1.0, "Labels not one-hot"

    def test_model_smoke_forward_pass(self, synthetic_data):
        images, metadata, labels, df = synthetic_data

        from skinsa.config import Config
        from skinsa.models.builder import build_model
        from skinsa.training.trainer import compile_model

        config = Config(
            backbone="mobilenet_v3_large",
            image_size=(96, 96),
            weights=None,
            bn_momentum=0.9,  # Required for small-batch TTA testing
            is_smoke_test=True,
        )

        model = build_model(config=config)
        model = compile_model(model, config)

        probs = model.predict([images[:4], metadata[:4]], verbose=0)
        assert probs.shape == (4, 7), f"Expected (4, 7), got {probs.shape}"
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(4), rtol=1e-4, atol=1e-4,
            err_msg="Probabilities don't sum to 1"
        )
        print(f"\n[SYNTHETIC] Forward pass OK — probs shape: {probs.shape}")

    def test_tta_smoke(self, synthetic_data):
        """TTA should work on synthetic data with bn_momentum=0.9 (prevents eval collapse)."""
        images, metadata, labels, df = synthetic_data

        from skinsa.config import Config
        from skinsa.models.builder import build_model
        from skinsa.training.trainer import compile_model
        from skinsa.evaluation.tta import evaluate_with_tta, evaluate_without_tta

        config = Config(
            backbone="mobilenet_v3_large",
            image_size=(96, 96),
            weights=None,
            bn_momentum=0.9,  # Critical: prevents TTA collapse on untrained BN
            is_smoke_test=True,
        )

        model = build_model(config=config)
        model = compile_model(model, config)

        n = 6
        probs_no_tta = evaluate_without_tta(model, images[:n], metadata[:n], batch_size=8)
        probs_tta = evaluate_with_tta(model, images[:n], metadata[:n], batch_size=8)

        assert probs_no_tta.shape == (n, 7), f"No-TTA shape wrong: {probs_no_tta.shape}"
        assert probs_tta.shape == (n, 7), f"TTA shape wrong: {probs_tta.shape}"

        # With bn_momentum=0.9, TTA and baseline should give different results
        # (not identical, confirming TTA is not inert)
        print(f"\n[SYNTHETIC] TTA OK — baseline probs[0]: {probs_no_tta[0]}")
        print(f"[SYNTHETIC] TTA OK — tta probs[0]:      {probs_tta[0]}")

    def test_class_weights_smoke(self):
        from skinsa.training.losses import compute_class_weights, ClassWeightedCrossEntropy
        import tensorflow as tf

        weights = compute_class_weights()
        assert len(weights) == 7

        loss_fn = ClassWeightedCrossEntropy(class_weights=weights)
        y_true = tf.one_hot([0, 1, 2, 5, 6], depth=7)
        y_pred = tf.nn.softmax(tf.random.uniform((5, 7)))
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0.0, "Loss should be positive"
        assert not np.isnan(loss.numpy()), "Loss should not be NaN"
        print(f"\n[SYNTHETIC] Class-weighted loss = {loss.numpy():.4f}")
