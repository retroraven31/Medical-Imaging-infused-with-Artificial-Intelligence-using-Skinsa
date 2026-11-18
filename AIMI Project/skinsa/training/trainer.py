"""
Training logic for skinsa.
Implements both experiment protocols:
- paper: stratified random 90/10 split
- lesion: StratifiedGroupKFold(10) grouped by lesion_id with TTA evaluation
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from skinsa.config import CLASS_NAMES, Config
from skinsa.models.builder import build_model
from skinsa.training.losses import ClassWeightedCrossEntropy, compute_class_weights


def compile_model(model, config: Config, class_weights: Optional[np.ndarray] = None):
    """
    Compile model with Adam(lr=1e-3, epsilon=0.1) and class-weighted cross-entropy.
    """
    if class_weights is None and config.use_class_weights:
        class_weights = compute_class_weights()

    loss_fn = ClassWeightedCrossEntropy(
        class_weights=class_weights if config.use_class_weights else None,
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=config.lr,
            epsilon=config.epsilon,
        ),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False, num_thresholds=200),
        ],
    )
    return model


def build_callbacks(config: Config, checkpoint_path: Path) -> list:
    """
    Build training callbacks:
    - ModelCheckpoint: best val_accuracy
    - ReduceLROnPlateau: factor=0.2, patience=10
    - EarlyStopping: patience=25 on val_accuracy
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.CSVLogger(
            str(checkpoint_path / "training_log.csv"),
            append=False,
        ),
    ]


def run_training(
    config: Config,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    class_weights: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Build, compile, and train the model on the given datasets.

    Returns
    -------
    model : trained Keras Model
    history : training history dict
    """
    if output_dir is None:
        output_dir = config.output_dir / f"{config.backbone}_{config.protocol}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config=config)
    model = compile_model(model, config, class_weights)

    model.summary(print_fn=lambda x: print(x, flush=True))

    cb_list = build_callbacks(config, output_dir)

    print(f"\n{'='*60}")
    print(f"[skinsa] TRAINING STARTED")
    print(f"  Protocol : {config.protocol}")
    print(f"  Backbone : {config.backbone}")
    print(f"  Epochs   : {config.epochs}")
    print(f"  Batch    : {config.batch_size}")
    print(f"  LR       : {config.lr}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=cb_list,
        verbose=2,
    )

    # Save history JSON
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\n[skinsa] Training complete. Best model saved to: {output_dir}")
    return model, history_dict
