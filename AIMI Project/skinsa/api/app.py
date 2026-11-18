"""
FastAPI backend for skinsa web application.
Provides endpoints for:
- Single-image diagnosis with soft-attention heatmap
- Batch CSV evaluation
- Results upload and reporting
- Experiment status tracking
"""

import base64
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Global app state
app = FastAPI(
    title="skinsa — Skin Lesion Classifier",
    description="Deep Learning Soft-Attention Model (Sensors 2022 replication)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state container
class AppState:
    model = None
    config = None
    backbone_name = "mobilenet_v3_large"
    model_loaded = False
    results_bundle = {}
    training_status = "idle"


state = AppState()

# Frontend directory search
CANDIDATE_DIRS = [
    Path(__file__).resolve().parent.parent.parent / "frontend",
    Path(__file__).resolve().parent.parent / "frontend",
    Path.cwd() / "frontend",
]
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
for d in CANDIDATE_DIRS:
    if d.exists() and (d / "index.html").exists():
        FRONTEND_DIR = d
        break

print(f"[skinsa] Serving frontend from: {FRONTEND_DIR} (exists: {FRONTEND_DIR.exists()})")


# ============================================================
# Utility helpers
# ============================================================

def load_or_get_model():
    """Load model lazily (once per server session)."""
    if state.model_loaded and state.model is not None:
        return state.model

    from skinsa.config import Config
    from skinsa.models.builder import build_model
    from skinsa.training.trainer import compile_model

    config = Config(
        backbone=state.backbone_name,
        image_size=(224, 224),
        weights=None,  # Weights not downloaded, smoke mode
        is_smoke_test=True,
    )
    model = build_model(config=config)
    state.model = model
    state.config = config
    state.model_loaded = True
    return model


def preprocess_image(pil_image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resize and normalize a PIL image to float32 (1, H, W, 3)."""
    img = pil_image.convert("RGB").resize(
        (target_size[1], target_size[0]), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # (1, H, W, 3)


def default_metadata_vector() -> np.ndarray:
    """Return a neutral 21-dim metadata vector for single-image inference."""
    from skinsa.data.metadata import encode_metadata_dict
    meta = encode_metadata_dict({"age": 50, "sex": "unknown", "localization": "unknown"})
    return meta[np.newaxis, :]  # (1, 21)


def array_to_b64_jpg(arr: np.ndarray, quality: int = 85) -> str:
    """Convert a uint8 image ndarray to base64-encoded JPEG string."""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()


# ============================================================
# Health / Info endpoints
# ============================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "model_loaded": state.model_loaded,
        "backbone": state.backbone_name,
    }


@app.get("/api/classes")
async def get_classes():
    from skinsa.config import CLASS_NAMES, CLASS_DESCRIPTIONS
    classes = []
    for name in CLASS_NAMES:
        info = CLASS_DESCRIPTIONS.get(name, {})
        classes.append({
            "id": name,
            "name": info.get("name", name),
            "category": info.get("category", "Unknown"),
            "risk": info.get("risk", "Unknown"),
            "color": info.get("color", "#666666"),
        })
    return {"classes": classes}


@app.get("/api/distribution")
async def get_distribution():
    from skinsa.config import HAM10000_DISTRIBUTION, TOTAL_HAM10000_SAMPLES
    data = []
    for cls, count in sorted(HAM10000_DISTRIBUTION.items(), key=lambda x: -x[1]):
        data.append({
            "class": cls,
            "count": count,
            "percentage": round(count / TOTAL_HAM10000_SAMPLES * 100, 2),
        })
    return {"distribution": data, "total": TOTAL_HAM10000_SAMPLES}


@app.get("/api/architecture")
async def get_architecture():
    from skinsa.config import BACKBONE_SPECS
    backbone = state.backbone_name
    spec = BACKBONE_SPECS.get(backbone, {})
    return {
        "backbone": backbone,
        "display_name": spec.get("display_name", backbone),
        "cut_layer": spec.get("cut_layer", "—"),
        "feature_channels": spec.get("feature_channels", 0),
        "feature_hw": spec.get("feature_hw", [7, 7]),
        "soft_attention_params": spec.get("soft_attention_params", 0),
        "total_trainable_params": spec.get("total_trainable_params", 0),
        "paper_metrics": __import__("skinsa.config", fromlist=["PAPER_REPORTED_METRICS"]).PAPER_REPORTED_METRICS.get(backbone, {}),
    }


@app.post("/api/set-backbone")
async def set_backbone(backbone: str = Form(...)):
    if backbone not in ["mobilenet_v3_large", "densenet201"]:
        raise HTTPException(status_code=400, detail=f"Unknown backbone: {backbone}")
    state.backbone_name = backbone
    state.model_loaded = False  # Force model reload
    state.model = None
    return {"status": "ok", "backbone": backbone}


# ============================================================
# Diagnosis endpoint
# ============================================================

@app.post("/api/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    age: Optional[str] = Form(None),
    sex: Optional[str] = Form("unknown"),
    localization: Optional[str] = Form("unknown"),
):
    """
    Single-image skin lesion diagnosis with soft-attention heatmap.

    Returns:
    - Predicted class with probability
    - Full 7-class probability distribution
    - Base64-encoded attention heatmap overlay
    - Inference time (ms)
    - Clinical note (pending GPU weights)
    """
    # Load model
    model = load_or_get_model()

    # Read and preprocess image
    raw_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(raw_bytes))
    img_arr = preprocess_image(pil_img)  # (1, H, W, 3)

    # Encode metadata
    from skinsa.data.metadata import encode_metadata_dict
    age_val = None
    if age and age.strip():
        try:
            age_val = float(age.strip())
        except ValueError:
            age_val = None

    meta_dict = {
        "age": age_val,
        "sex": sex or "unknown",
        "localization": localization or "unknown",
    }
    meta_arr = encode_metadata_dict(meta_dict)[np.newaxis, :]

    # Time inference
    t0 = time.perf_counter()
    probs = model.predict([img_arr, meta_arr], verbose=0)[0]
    inference_ms = (time.perf_counter() - t0) * 1000.0

    from skinsa.config import CLASS_NAMES, CLASS_DESCRIPTIONS

    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    cls_info = CLASS_DESCRIPTIONS.get(pred_cls, {})
    class_probs = [
        {
            "class": c,
            "name": CLASS_DESCRIPTIONS.get(c, {}).get("name", c),
            "probability": float(probs[i]),
            "color": CLASS_DESCRIPTIONS.get(c, {}).get("color", "#666666"),
        }
        for i, c in enumerate(CLASS_NAMES)
    ]

    # Generate attention heatmap
    heatmap_b64 = None
    try:
        from skinsa.models.builder import get_attention_extractor
        import tensorflow as tf

        # Try to get attention map
        sa_layer = model.get_layer("soft_attention")
        backbone_model = model.get_layer(f"{state.backbone_name}_cut")
        feature_map = backbone_model.predict(img_arr, verbose=0)
        attn_features, attn_map = sa_layer(feature_map, return_attention=True)

        attn_np = attn_map.numpy()[0]  # (H, W, 1)

        from skinsa.plots import overlay_attention_heatmap
        orig_img_np = (img_arr[0] * 255).astype(np.uint8)
        overlay = overlay_attention_heatmap(img_arr[0], attn_np, alpha=0.60)
        heatmap_b64 = array_to_b64_jpg(overlay)
    except Exception as e:
        heatmap_b64 = None

    # Encode original image as thumbnail
    orig_pil = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    orig_pil.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    orig_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()

    return JSONResponse({
        "prediction": {
            "class": pred_cls,
            "name": cls_info.get("name", pred_cls),
            "confidence": round(pred_conf * 100, 2),
            "risk": cls_info.get("risk", "Unknown"),
            "category": cls_info.get("category", "Unknown"),
        },
        "probabilities": sorted(class_probs, key=lambda x: -x["probability"]),
        "metadata_used": meta_dict,
        "inference_ms": round(inference_ms, 2),
        "heatmap": heatmap_b64,
        "original_image": orig_b64,
        "disclaimer": (
            "⚠️ RESEARCH PROTOTYPE ONLY — This model was initialized without "
            "trained weights. Predictions are NOT clinically meaningful. "
            "Real results require GPU training on HAM10000. See documentation."
        ),
        "model_info": {
            "backbone": state.backbone_name,
            "weights_loaded": False,
            "is_trained": False,
        },
    })


# ============================================================
# Results upload endpoint
# ============================================================

@app.post("/api/upload-results")
async def upload_results(results_zip: UploadFile = File(...)):
    """
    Upload a results_bundle.zip from a completed GPU training run.
    Populates the comparison table with real metrics.
    """
    raw = await results_zip.read()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if "results.json" not in names:
                raise HTTPException(
                    status_code=400,
                    detail=f"results_bundle.zip must contain 'results.json'. Found: {names}"
                )
            results_json = json.loads(zf.read("results.json").decode("utf-8"))
            state.results_bundle = results_json
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")

    return {"status": "ok", "loaded_keys": list(state.results_bundle.keys())}


@app.get("/api/results")
async def get_results():
    """Return experiment results (PENDING if no GPU run uploaded yet)."""
    from skinsa.config import PAPER_REPORTED_METRICS, BACKBONE_SPECS

    backbone = state.backbone_name
    paper_metrics = PAPER_REPORTED_METRICS.get(backbone, {})

    if state.results_bundle:
        replicated = state.results_bundle.get("baseline", {})
        enhanced = state.results_bundle.get("enhanced", {})
        completed = True
    else:
        replicated = {}
        enhanced = {}
        completed = False

    return {
        "backbone": backbone,
        "training_completed": completed,
        "paper_metrics": paper_metrics,
        "replicated_metrics": replicated,
        "enhanced_metrics": enhanced,
        "status": (
            "Real GPU metrics loaded from results_bundle.zip"
            if completed else
            "🔴 PENDING — No GPU training results uploaded yet. Run on Colab T4 and upload results_bundle.zip"
        ),
    }


# ============================================================
# Serve frontend (SPA)
# ============================================================

def find_index_html() -> Optional[Path]:
    candidates = [
        FRONTEND_DIR / "index.html",
        Path("frontend/index.html"),
        Path("skinsa/frontend/index.html"),
        Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html",
        Path(__file__).resolve().parent.parent / "frontend" / "index.html",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

if FRONTEND_DIR.exists() and (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    p = find_index_html()
    if p is not None:
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>skinsa frontend not found</h1><p>Run from project root.</p>", status_code=404)

@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_spa(path: str):
    # Check if a static file is directly requested
    candidates = [
        FRONTEND_DIR / path,
        Path("frontend") / path,
        Path("skinsa/frontend") / path,
    ]
    for fp in candidates:
        if fp.exists() and fp.is_file():
            return FileResponse(str(fp))
    # Fallback to index.html for SPA client-side routing
    p = find_index_html()
    if p is not None:
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(f"<h1>404 — {path} not found</h1>", status_code=404)
