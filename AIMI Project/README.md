# skinsa: Skin Lesion Classification on Imbalanced Data Using Deep Learning with Soft Attention

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0%2B-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Replication and enhancement of the research paper:
> **Nguyen, V. D., Bui, N. D., & Do, H. K. (2022).**  
> *Skin Lesion Classification on Imbalanced Data Using Deep Learning with Soft Attention.*  
> **Sensors**, 22(19), 7530. [DOI: 10.3390/s22197530](https://doi.org/10.3390/s22197530)

---

## 🔬 Architectural Summary

The model replicates the authors' exact network topology on the **HAM10000** dermoscopy dataset (10,015 images across 7 diagnostic classes, where Melanocytic Nevi makes up ~67% of all data):

1. **Backbone Cut:**
   - **MobileNetV3Large:** Cut right after `expanded_conv_14_expand_bn` (activation layer), yielding a $7 \times 7$ feature map with **960 channels**.
   - **DenseNet201:** Cut at `conv5_block29_1_relu`, yielding a $7 \times 7$ feature map with **128 channels**.
2. **16-Head Soft-Attention Layer:**
   - Ported to TensorFlow ops using the authors' Conv3D layout.
   - Kernel shape: `(C, 3, 3, 1, 16)`, stride `(1, 1, C)`, no learnable $\gamma$.
   - **Parameter counts:**
     - MobileNetV3Large ($C=960$): **138,256** parameters ($960 \times 3 \times 3 \times 1 \times 16 + 16$).
     - DenseNet201 ($C=128$): **18,448** parameters ($128 \times 3 \times 3 \times 1 \times 16 + 16$).
3. **Feature Pooling & Concatenation:**
   - Max-pools both the plain and attended feature maps (`MaxPooling2D(pool_size=2, strides=2, padding='same')` $\to 4 \times 4 \times C$).
   - Concatenates along channels $\to 4 \times 4 \times 2C$, followed by ReLU, Dropout(0.5), and Flatten:
     - MobileNetV3Large visual vector: **30,720** elements ($4 \times 4 \times 1920$).
     - DenseNet201 visual vector: **4,096** elements ($4 \times 4 \times 256$).
4. **Metadata Branch:**
   - Dense(4096) layer over **21 patient metadata features** (age, sex, body site) $\to$ **90,112** parameters ($21 \times 4096 + 4096$).
5. **Multi-Modal Fusion & Softmax Output:**
   - Concatenates visual vector ($30,720$ or $4,096$) + metadata vector ($4,096$).
   - Dense(7) with softmax activation:
     - MobileNet head: **243,719** parameters ($34,816 \times 7 + 7$).
     - DenseNet head: **57,351** parameters ($8,192 \times 7 + 7$).
6. **Total Trainable Parameters:**
   - **MobileNetV3Large:** **2,646,679**
   - **DenseNet201:** **17,382,935**

---

## ⚖️ Imbalance Mitigation: Class-Weighted Loss

To counterbalance the extreme class imbalance in HAM10000 (NV = 6,705 samples vs. DF = 115 samples, a 58:1 ratio), training uses inverse class-frequency cross-entropy:

$$w_i = \frac{N}{C \cdot N_i}$$

where $N = 10,015$, $C = 7$, and $N_i$ is the count of class $i$.

---

## 🧪 Protocols: Baseline vs. Enhancement

| Feature | Baseline (`--protocol paper`) | Enhanced (`--protocol lesion`) |
| :--- | :--- | :--- |
| **Splitting Strategy** | Stratified random 90/10 split | `StratifiedGroupKFold(10)` grouped by `lesion_id` (80/10/10) |
| **Lesion Leakage** | Possible (same patient lesion in train & test) | **Strictly 0%** (zero patient overlap) |
| **Evaluation Mode** | Standard single-view inference | **8-view Dihedral TTA** ($D_4$ symmetry average) |
| **Weights** | Shared trained weights | **Identical weights** (isolating TTA gain) |

---

## 📊 Comparison Table

| Metric | Sensors 2022 (Reported) | Replicated (`--protocol paper`) | Enhanced (`--protocol lesion` + TTA) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.8600 | *Results Pending (GPU run)* | *Results Pending (GPU run)* |
| **Precision (Macro)** | 0.7600 | *Results Pending (GPU run)* | *Results Pending (GPU run)* |
| **Recall (Macro)** | 0.7700 | *Results Pending (GPU run)* | *Results Pending (GPU run)* |
| **F1-Score (Macro)** | 0.7600 | *Results Pending (GPU run)* | *Results Pending (GPU run)* |
| **AUC (Macro OVR)** | 0.9650 | *Results Pending (GPU run)* | *Results Pending (GPU run)* |
| **Inference Latency** | ~28.5 ms | *Results Pending (GPU run)* | *Results Pending (GPU run)* |

> **Note:** Real HAM10000 metrics require execution on GPU (Colab T4 or local GPU). Synthetic smoke runs are strictly kept out of scientific comparison tables. Upload `results_bundle.zip` into the Web UI to populate real measured numbers.

---

## 🚀 Quickstart & Installation

```bash
# Clone and install dependencies
git clone <repo-url> skinsa
cd skinsa
pip install -e .
```

### Running the Tests (9 Fidelity Criteria)
```bash
python -m pytest tests/test_fidelity.py -v
python -m pytest tests/test_smoke.py -v
```

### Running CPU Smoke Test on Synthetic Data (96 px)
```bash
skinsa --smoke-test --backbone mobilenet_v3_large --bn-momentum 0.9
```

### Running the Web UI & API Server
```bash
python run_server.py
# Open http://localhost:8000 in your browser
```

---

## 🌐 Web UI / UX Frontend Features

- **Interactive Dermoscopy Diagnosis:** Drag-and-drop lesion images, input age/sex/localization.
- **Explainable Soft-Attention Heatmaps:** Real-time visual overlay revealing high-attention lesion regions.
- **8-View Dihedral TTA Inspector:** Visualizes all $D_4$ symmetries and ensemble probability convergence.
- **Results Bundle Importer:** Instant drag-and-drop import of `results_bundle.zip` from Colab GPU runs to populate comparison curves and confusion matrices.
