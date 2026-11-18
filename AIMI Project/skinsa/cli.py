"""
Command-line interface for skinsa.
Runs both protocols (paper/lesion) with comprehensive evaluation and report generation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="skinsa",
        description=(
            "Skin Lesion Classification on Imbalanced Data Using Deep Learning "
            "with Soft Attention — Sensors 2022 replication\n\n"
            "PROTOCOLS:\n"
            "  paper  : Stratified random 90/10 split (authors' baseline)\n"
            "  lesion : StratifiedGroupKFold(10) grouped by lesion_id + 8-view TTA"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ham10000"),
        help="Directory containing HAM10000 dataset (default: data/ham10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for all outputs (default: outputs)",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        choices=["mobilenet_v3_large", "densenet201"],
        default="mobilenet_v3_large",
        help="Backbone architecture (default: mobilenet_v3_large)",
    )
    parser.add_argument(
        "--protocol",
        choices=["paper", "lesion"],
        default="paper",
        help="Experiment protocol (default: paper)",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Pretrained weights (default: imagenet)",
    )
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=0.99,
        help="BatchNorm momentum (use 0.9 for small-batch smoke tests)",
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=250, help="Max epochs (default: 250)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class-weighted loss")

    # Protocol modifiers
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick CPU smoke test on synthetic data at 96px")
    parser.add_argument("--results-bundle", type=Path, default=None,
                        help="Path to results_bundle.zip from a completed GPU run")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return parser.parse_args(argv)


def run_smoke_test(args) -> int:
    """Run CPU smoke test on synthetic 96px data with clear banners."""
    from skinsa.config import Config
    from skinsa.data.synthetic import create_synthetic_dataset
    from skinsa.data.loader import build_tf_dataset_from_arrays
    from skinsa.models.builder import build_model
    from skinsa.training.trainer import compile_model
    from skinsa.evaluation.metrics import compute_metrics

    print("\n" + "=" * 70)
    print("⚠️  SYNTHETIC SMOKE TEST — NOT REAL RESULTS")
    print("    Data: randomly generated, 96×96 pixels, CPU-only")
    print("    Purpose: verify pipeline runs end-to-end")
    print("=" * 70 + "\n")

    config = Config(
        backbone=args.backbone,
        image_size=(96, 96),
        weights=None,
        bn_momentum=float(args.bn_momentum),
        is_smoke_test=True,
        protocol=args.protocol,
        epochs=min(args.epochs, 3),
        use_class_weights=not args.no_class_weights,
        random_state=args.seed,
    )

    images, metadata, labels, df = create_synthetic_dataset(
        num_samples=100, size=(96, 96), seed=args.seed
    )

    split = int(0.9 * len(images))
    train_ds = build_tf_dataset_from_arrays(
        images[:split], metadata[:split], labels[:split],
        batch_size=16, shuffle=True, is_training=True
    )
    val_ds = build_tf_dataset_from_arrays(
        images[split:], metadata[split:], labels[split:],
        batch_size=16, shuffle=False, is_training=False
    )

    model = build_model(config=config)
    model = compile_model(model, config)

    print(f"\n[SMOKE] Model total parameters: {model.count_params():,}")
    print("[SMOKE] Starting 3-epoch training on SYNTHETIC data...\n")

    history = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    print("\n" + "=" * 70)
    print("⚠️  SMOKE TEST COMPLETE — SYNTHETIC METRICS ONLY (NOT REAL)")
    for metric, vals in history.history.items():
        print(f"  {metric}: {vals[-1]:.4f}")
    print("=" * 70 + "\n")
    return 0


def run_full_training(args) -> int:
    """Run full training on real HAM10000 data."""
    from skinsa.config import Config
    from skinsa.data.loader import load_ham10000_metadata, build_tf_dataset
    from skinsa.data.splits import split_paper_protocol, split_lesion_protocol, verify_no_lesion_leakage
    from skinsa.models.builder import build_model
    from skinsa.training.trainer import compile_model, build_callbacks, run_training
    from skinsa.evaluation.metrics import compute_metrics, compute_confusion_matrix, compute_roc_curves, measure_inference_time
    from skinsa.evaluation.tta import evaluate_with_tta, evaluate_without_tta
    from skinsa.plots import (
        plot_confusion_matrix, plot_roc_curves, plot_training_history, generate_comparison_report
    )

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        weights=args.weights if args.weights != "none" else None,
        bn_momentum=args.bn_momentum,
        protocol=args.protocol,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_class_weights=not args.no_class_weights,
        random_state=args.seed,
    )

    print(f"\n[skinsa] Loading HAM10000 metadata from {config.data_dir}")
    df = load_ham10000_metadata(config.data_dir)
    print(f"[skinsa] Loaded {len(df)} samples, {df['dx'].nunique()} classes")

    output_dir = config.output_dir / f"{config.backbone}_{config.protocol}"
    image_dir = config.data_dir

    if config.protocol == "paper":
        train_df, val_df = split_paper_protocol(df, random_state=config.random_state)
        print(f"[paper] Train: {len(train_df)}, Val: {len(val_df)}")
        train_ds = build_tf_dataset(train_df, image_dir, config.image_size, config.batch_size, shuffle=True, is_training=True)
        val_ds = build_tf_dataset(val_df, image_dir, config.image_size, config.batch_size, shuffle=False, is_training=False)

        model, history = run_training(config, train_ds, val_ds, output_dir=output_dir)

        # Evaluate baseline (no TTA)
        from skinsa.data.loader import load_image_to_array
        from skinsa.data.metadata import encode_metadata_dataframe
        from skinsa.config import CLASS_NAMES

        val_images = np.stack([load_image_to_array(image_dir / f"{row.image_id}.jpg", config.image_size) for _, row in val_df.iterrows()])
        val_meta = encode_metadata_dataframe(val_df)
        y_true = np.array([CLASS_NAMES.index(c) for c in val_df["dx"]])

        probs_baseline = evaluate_without_tta(model, val_images, val_meta, batch_size=config.batch_size)
        metrics_baseline = compute_metrics(y_true, probs_baseline)

    elif config.protocol == "lesion":
        train_df, val_df, test_df = split_lesion_protocol(df, random_state=config.random_state)
        clean, leaked = verify_no_lesion_leakage(train_df, test_df)
        print(f"[lesion] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"[lesion] Leakage check: {'CLEAN ✓' if clean else f'LEAKED: {leaked}'}")

        train_ds = build_tf_dataset(train_df, image_dir, config.image_size, config.batch_size, shuffle=True, is_training=True)
        val_ds = build_tf_dataset(val_df, image_dir, config.image_size, config.batch_size, shuffle=False, is_training=False)

        model, history = run_training(config, train_ds, val_ds, output_dir=output_dir)

    # Plots
    plot_training_history(history, output_dir / "training_history.png", title=f"Training History: {config.backbone} ({config.protocol})")
    print(f"[skinsa] Training plots saved to {output_dir}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          skinsa — Skin Lesion Classification              ║
║   Replication of Nguyen et al., Sensors 2022              ║
╚═══════════════════════════════════════════════════════════╝
  Backbone : {args.backbone}
  Protocol : {args.protocol}
  Weights  : {args.weights}
  Smoke    : {args.smoke_test}
""")

    import numpy as np  # noqa: needed for evaluate functions

    if args.smoke_test:
        return run_smoke_test(args)
    else:
        return run_full_training(args)


if __name__ == "__main__":
    sys.exit(main())
