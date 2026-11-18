"""
Visualization, heatmap and report generation module for skinsa.
Produces:
1. Confusion matrices
2. ROC curves
3. Soft-attention heatmaps overlaid on input images
4. Training history plots
5. Comparison tables
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless mode for server rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from skinsa.config import CLASS_NAMES, CLASS_DESCRIPTIONS


def plot_confusion_matrix(
    cm_data: Dict,
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> str:
    """
    Plot confusion matrix as a heatmap.
    Returns path to saved figure.
    """
    matrix = np.array(cm_data["matrix"])
    class_names = cm_data.get("class_names", CLASS_NAMES)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap=cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells
    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_roc_curves(
    roc_data: Dict,
    output_path: Path,
    title: str = "ROC Curves (OVR)",
) -> str:
    """Plot one-vs-rest ROC curves for all classes."""
    curves = roc_data.get("curves", {})

    # Build 7-class color palette matching CLASS_DESCRIPTIONS
    colors = [CLASS_DESCRIPTIONS.get(c, {}).get("color", "#666666") for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance")

    for idx, (cls_name, curve_data) in enumerate(curves.items()):
        color_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else idx % len(colors)
        ax.plot(
            curve_data["fpr"],
            curve_data["tpr"],
            color=colors[color_idx],
            lw=2.0,
            label=f"{cls_name} (AUC = {curve_data['auc']:.4f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_training_history(
    history_dict: Dict,
    output_path: Path,
    title: str = "Training History",
) -> str:
    """Plot train/val accuracy and loss curves from Keras history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    axes[0].plot(history_dict.get("accuracy", []), label="Train Accuracy", color="#3B82F6", lw=2)
    axes[0].plot(history_dict.get("val_accuracy", []), label="Val Accuracy", color="#10B981", lw=2)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Accuracy Curves", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Loss subplot
    axes[1].plot(history_dict.get("loss", []), label="Train Loss", color="#EF4444", lw=2)
    axes[1].plot(history_dict.get("val_loss", []), label="Val Loss", color="#F59E0B", lw=2)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Loss Curves", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def overlay_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Overlay a 2D attention map on an RGB image using a Jet colormap.

    Parameters
    ----------
    image : np.ndarray of shape (H, W, 3) in range [0, 1] float32.
    attention_map : np.ndarray of shape (H_attn, W_attn, 1) or (H_attn, W_attn).
    alpha : float
        Blending weight for the attention overlay (0=image, 1=heatmap).

    Returns
    -------
    overlay : np.ndarray of shape (H, W, 3) uint8.
    """
    from PIL import Image

    if attention_map.ndim == 3:
        attention_map = attention_map[..., 0]

    h, w = image.shape[:2]

    # Normalize attention map
    a_min, a_max = attention_map.min(), attention_map.max()
    if a_max > a_min:
        attn_norm = (attention_map - a_min) / (a_max - a_min)
    else:
        attn_norm = np.zeros_like(attention_map)

    # Upscale attention to image size
    attn_pil = Image.fromarray((attn_norm * 255).astype(np.uint8), mode="L")
    attn_pil = attn_pil.resize((w, h), Image.BILINEAR)
    attn_upscaled = np.array(attn_pil) / 255.0

    # Apply Jet colormap to get RGB heatmap
    heatmap_rgb = cm.jet(attn_upscaled)[:, :, :3]  # (H, W, 3) in [0, 1]

    # Blend with original image
    image_clamped = np.clip(image, 0, 1)
    blended = (1 - alpha) * image_clamped + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 1)

    return (blended * 255).astype(np.uint8)


def save_attention_heatmap_grid(
    images: List[np.ndarray],
    attention_maps: List[np.ndarray],
    class_names: List[str],
    output_path: Path,
    title: str = "Soft-Attention Heatmaps",
) -> str:
    """Plot a grid of original images + attention overlay side by side."""
    n = min(len(images), 6)  # Show up to 6 examples
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i]
        attn = attention_maps[i]
        overlay = overlay_attention_heatmap(img, attn, alpha=0.65)

        axes[i][0].imshow(np.clip(img, 0, 1))
        axes[i][0].set_title(f"Original ({class_names[i]})", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay)
        axes[i][1].set_title("Attention Heatmap", fontsize=9)
        axes[i][1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_comparison_report(
    results: Dict,
    output_path: Path,
    is_synthetic: bool = False,
) -> str:
    """
    Generate a markdown comparison report:
    - Paper reported vs. replicated vs. enhanced numbers.
    - Marks all results as PENDING until from actual runs.
    - Clearly banners any synthetic data results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# skinsa: Experiment Comparison Report\n"]
    lines.append("**Replication of:** Nguyen VD, Bui ND, Do HK. *Skin Lesion Classification on "
                 "Imbalanced Data Using Deep Learning with Soft Attention.* "
                 "Sensors 2022, 22(19), 7530. DOI: 10.3390/s22197530\n")

    if is_synthetic:
        lines.append("\n> [!WARNING] **SYNTHETIC DATA RESULTS** — These numbers are from CPU smoke "
                     "tests on randomly generated images and are **not comparable to any real metric**. "
                     "They serve only to verify the pipeline runs end-to-end.\n")

    lines.append("\n## Metric Comparison Table\n")
    lines.append("| Metric | Paper (Reported) | Replicated (Baseline) | Enhanced (Lesion+TTA) |")
    lines.append("|--------|-----------------|----------------------|----------------------|")

    # Paper reported numbers
    backbone = results.get("backbone", "mobilenet_v3_large")
    from skinsa.config import PAPER_REPORTED_METRICS
    paper = PAPER_REPORTED_METRICS.get(backbone, {})
    baseline = results.get("baseline", {})
    enhanced = results.get("enhanced", {})

    def fmt(v, digits=4):
        if v is None or (isinstance(v, float) and (v != v)):
            return "🔴 PENDING"
        return f"{v:.{digits}f}"

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        paper_v = paper.get(metric)
        base_v = baseline.get(metric)
        enh_v = enhanced.get(metric)
        lines.append(f"| {metric.capitalize()} | {fmt(paper_v)} | {fmt(base_v)} | {fmt(enh_v)} |")

    exec_ms = baseline.get("inference_time_ms")
    lines.append(f"| Inference (ms/img) | {fmt(paper.get('execution_time_ms'))} | {fmt(exec_ms)} | — |")

    lines.append("\n## Execution Notes\n")
    if not results.get("training_completed", False):
        lines.append("🔴 **Training not completed** — No GPU run has produced real metrics yet. "
                     "All replicated/enhanced results above are PENDING.\n")
        lines.append("**To obtain real results:**\n")
        lines.append("1. Upload HAM10000 dataset to Google Colab (Colab notebook provided)\n")
        lines.append("2. Run: `skinsa --protocol paper --backbone mobilenet_v3_large`\n")
        lines.append("3. Upload `results_bundle.zip` to this app to populate the table\n")
    else:
        lines.append(f"✅ Training completed. Protocol: `{results.get('protocol', 'unknown')}`, "
                     f"Backbone: `{backbone}`\n")

    lines.append("\n## Architecture Summary\n")
    from skinsa.config import BACKBONE_SPECS
    spec = BACKBONE_SPECS.get(backbone, {})
    lines.append(f"- **Backbone:** {spec.get('display_name', backbone)} cut at `{spec.get('cut_layer', '—')}`\n")
    lines.append(f"- **Feature map:** 7×7 × {spec.get('feature_channels', '?')} channels\n")
    lines.append(f"- **Soft-Attention params:** {spec.get('soft_attention_params', '?'):,}\n")
    lines.append(f"- **Total trainable params:** {spec.get('total_trainable_params', '?'):,}\n")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    return str(output_path)
