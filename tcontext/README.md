# `tcontext` — Core Pipeline Reference

End-to-end deep learning and retrieval experiment for binary image classification (Cats vs Dogs).
Uses 25% class-balanced sampling with explicit anti-overfitting controls.

> For running the app, see the [root README](../README.md). This document covers the internals.

---

## File Reference

| File | Purpose |
|---|---|
| `web_app.py` | Streamlit dashboard. Entry point for the interactive demo. |
| `quick500_experiment.py` | Full pipeline: sampling → EDA → training → retrieval benchmark → reports. |
| `comparative_eda.py` | Generates comparative method metrics plot and EDA markdown report. |
| `query_demo.py` | CLI tool for querying the vector DB and adding new images to memory. |
| `dataset_utils.py` | Dataset discovery, validation, RGB cleaning, and local cache management. |
| `requirements.txt` | Pinned dependencies for the entire project. |

---

## Path Resolution

All scripts use `Path(__file__).resolve().parent` to locate project files. They work correctly regardless of the working directory they are called from — repo root, `tcontext/`, or any other path.

`web_app.py` additionally calls `os.chdir(APP_DIR)` at import time so that subprocess calls launched from within Streamlit also resolve paths correctly.

---

## Dataset Preparation

### Source
Microsoft Cats vs Dogs (Kaggle PetImages archive), accessed via the local TensorFlow Datasets download cache.

### Pipeline (`dataset_utils.py`)
1. Locates the most recent `cats_vs_dogs` zip in the TFDS download cache.
2. Extracts `PetImages/Cat` and `PetImages/Dog` to `data/raw/PetImages/`.
3. Validates each image (skips corrupted files), converts to RGB, re-saves as JPEG.
4. Writes a clean cache to `data/clean/` with up to 2500 images per class.
5. A `.ready_rgb_v2_2500` marker prevents redundant re-processing on subsequent runs.

### Sampling (in `quick500_experiment.py`)
- 25% stratified random sample per class → **625 cats + 625 dogs = 1250 total**
- 80/20 train/val split → **1000 train, 250 validation**
- Seeded with `--seed` for reproducibility

---

## Modeling

### Deep Learning Classifier
- Backbone: `EfficientNetB0` (ImageNet pretrained, fully frozen)
- Head: `GlobalAveragePooling2D → Dropout(0.45) → Dense(1, sigmoid, L2=1e-4)`
- Optimizer: `Adam(lr=3e-4)` | Loss: `BinaryCrossentropy(label_smoothing=0.05)`
- Input: `224×224`, batch size `32`
- Augmentation (train only): random horizontal flip, rotation ±8%, zoom ±12%
- Callbacks: `EarlyStopping(patience=3)`, `ReduceLROnPlateau(factor=0.35, patience=1)`

### Retrieval System
- Encoder: `CLIP ViT-B/32` (`openai/clip-vit-base-patch32`) — frozen, no training
- Vector store: `ChromaDB` persistent collection, cosine similarity space
- Inference: top-5 nearest neighbour search → majority-vote label

---

## Intelligent Memory Orchestrator (IMO)

Implemented in `web_app.py` via `_passes_memory_threshold()`. Before any image is added to the vector DB through the web UI:

1. The EfficientNet classifier scores the image.
2. If the predicted label **disagrees** with the user-supplied label → rejected (conflict detection).
3. If confidence is **below 75%** → rejected (confidence gate).
4. Only images passing both checks are embedded and inserted into ChromaDB.

This keeps the retrieval memory clean without any manual curation.

---

## Key Output Files

| Path | Content |
|---|---|
| `artifacts/sampled25_summary.json` | Full metrics, timing, compute profile — consumed by the dashboard |
| `artifacts/sampled25_comparison_metrics.csv` | Side-by-side classifier vs retrieval metric table |
| `artifacts/sampled25_training_history.csv` | Per-epoch loss/accuracy/AUC values |
| `artifacts/training_curves.png` | Loss, accuracy, AUC curves across epochs |
| `artifacts/classifier_roc_curve.png` | ROC curve with AUC annotation |
| `artifacts/comparison_confusion_matrices.png` | Classifier and retrieval confusion matrices |
| `artifacts/comparative_method_metrics.png` | Bar chart comparing accuracy/precision/recall/F1 |
| `artifacts/comparative_learning_curve.png` | Train vs val accuracy convergence plot |
| `model/cats_vs_dogs_model_quick500.keras` | Saved Keras model |
| `reports/sampled25_report.md` | Full experiment report (auto-generated) |
| `reports/comparative_eda_report.md` | Comparative EDA narrative (auto-generated) |
| `vector_db/` | Persistent ChromaDB collections |

---

## Latest Experiment Snapshot (Seed 777)

| Metric | Classifier (EfficientNetB0) | Retrieval (CLIP + ChromaDB) |
|---|---|---|
| Accuracy | 0.9800 | 0.9880 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9600 | 0.9760 |
| F1 | 0.9796 | 0.9879 |

- Epochs completed: 12 | Best epoch (val_loss): 12
- No overfitting gap > 0.08 detected across any epoch

---

## Reproducibility Notes

- `--seed` controls both the random subset sampling and the train/val split shuffle.
- The clean cache (`data/clean/`) persists across runs. Delete it to force a full re-clean.
- Each experiment run creates a new seed-stamped vector DB (`vector_db/clip_chroma_db_seed_<N>/`). The app always picks the most recently modified one.
- TFDS zip must be present locally before the first run. Download once with: `import tensorflow_datasets as tfds; tfds.load("cats_vs_dogs")`.
