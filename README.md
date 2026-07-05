# A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning

This project implements a novel framework for continuous, incremental learning **without fine-tuning model weights**. An **Intelligent Memory Orchestrator (IMO)** combined with an external vector database (ChromaDB) provides near-real-time knowledge integration, replacing the costly backpropagation loop with a fast embedding-and-retrieve cycle.

---

## 📖 Abstract

Modern multi-modal models achieve strong performance but struggle with continuous knowledge integration. Traditional adaptation depends on fine-tuning, which introduces high computational costs, risks of catastrophic forgetting, and slow deployment cycles.

This thesis proposes a **non-parametric, retrieval-centric framework**. Instead of fine-tuning the backbone network, new data is encoded into embeddings and stored in an external vector memory (ChromaDB). A dedicated IMO layer controls memory quality via confidence gating, prototype consistency, and outlier filtering. During inference, retrieval evidence is dynamically fused with the base model to form the final prediction.

---

## 🗂️ Repository Structure

```
Thesis/
├── launch_app.ps1                  ← One-click launcher (start here)
├── run_demo.ps1                    ← Full pipeline + launch (runs experiment first if needed)
├── tcontext/
│   ├── web_app.py                  ← Streamlit dashboard
│   ├── quick500_experiment.py      ← Full training + retrieval benchmark pipeline
│   ├── comparative_eda.py          ← Generates comparative EDA plots and report
│   ├── query_demo.py               ← CLI tool for incremental memory updates
│   ├── dataset_utils.py            ← Dataset discovery, cleaning, caching
│   ├── requirements.txt            ← Pinned Python dependencies
│   ├── artifacts/                  ← Plots, CSVs, JSON summary
│   ├── data/                       ← Raw, clean, and sampled image datasets
│   ├── model/                      ← Trained Keras model (.keras)
│   ├── reports/                    ← Markdown experiment reports
│   └── vector_db/                  ← Persistent ChromaDB vector store
└── logs/                           ← Execution logs
```

---

## ⚡ Core Concepts

### 1. Separation of Memory
- **Parametric Memory**: A frozen encoder (`EfficientNetB0` or `CLIP`) that extracts rich feature embeddings — weights never change.
- **Non-parametric Memory**: A dynamic external vector database storing embeddings, labels, and quality scores — updated in milliseconds.

### 2. Intelligent Memory Orchestrator (IMO)
New data passes through a quality gate before entering memory:
- **Confidence Gating**: rejects samples where the classifier confidence is below 75%.
- **Conflict Detection**: quarantines noisy labels or samples that disagree with the existing class prototype.
- **Temporal Weighting**: prevents stale information from skewing new predictions.

---

## 🚀 Getting Started

### Prerequisites

Python 3.10+ is required. Install all dependencies from the pinned requirements file:

```powershell
pip install -r tcontext\requirements.txt
```

> All packages are pinned to the exact versions used during development. See `tcontext/requirements.txt` for the full list.

---

## ▶️ Running the App — Two Options

### Option 1: One-Click Launch (recommended)

The fastest path. Use this when the experiment artifacts (model, vector DB) are already present.

```powershell
.\launch_app.ps1
```

- Kills any existing process on port 8501.
- Starts Streamlit and opens `http://localhost:8501` in your browser automatically.
- Keeps the terminal window open so you can see live logs.

### Option 2: Full Pipeline + Launch

Use this on a fresh clone, or when you want to regenerate all experiment artifacts before opening the dashboard.

```powershell
.\run_demo.ps1
```

- Checks whether the trained model and vector DB already exist.
- If missing, runs `quick500_experiment.py --seed 777` (training + retrieval benchmark) then `comparative_eda.py`.
- Launches the Streamlit dashboard once assets are ready.

> The first run trains EfficientNetB0 and builds the CLIP vector index — this takes several minutes depending on your hardware. Subsequent runs skip straight to the dashboard.

---

## 🖥️ Using the Dashboard

| Section | What you can do |
|---|---|
| **Overview** | See side-by-side accuracy, F1, and timing metrics for the classifier vs retrieval system. Initialize or rebuild the retrieval memory. |
| **Comparative EDA** | View training convergence curves and the full comparative method metrics plot. |
| **Artifacts** | Browse all generated plots (confusion matrices, ROC curve, training curves) and raw CSV metrics. |
| **Live Demo** | Upload an image (or pick a built-in sample) and get simultaneous predictions from EfficientNet and CLIP+VectorDB. Upload 1–10 new labeled images to update retrieval memory **instantly, without retraining**. |

---

## 🖧 Standalone CLI Commands

All commands run from the **repository root**.

**Run the full training + retrieval benchmark:**
```powershell
python tcontext\quick500_experiment.py --seed 777
```

**Capture full epoch logs to file:**
```powershell
python tcontext\quick500_experiment.py --seed 777 2>&1 | Tee-Object -FilePath logs\run_sampled25_terminal.log
```

**Generate comparative EDA report and plots:**
```powershell
python tcontext\comparative_eda.py
```

**Query an image against the vector DB:**
```powershell
python tcontext\query_demo.py --query-image "path\to\image.jpg" --top-k 5
```

**Add a new image to retrieval memory instantly:**
```powershell
python tcontext\query_demo.py --add-image "path\to\new_dog.jpg" --label dogs
```

---

## 📊 Experimental Results (Seed 777, 25% sample)

Tested on the Microsoft Cats vs Dogs dataset. 1250 total images (625 cats / 625 dogs), 80/20 train/val split.

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| EfficientNetB0 Classifier | 0.9800 | 1.0000 | 0.9600 | 0.9796 |
| CLIP + ChromaDB Retrieval | 0.9880 | 1.0000 | 0.9760 | 0.9879 |

Key takeaway: the retrieval system **matches or exceeds** classifier accuracy while supporting **O(1) incremental updates** — no retraining, no forgetting risk.

Full reports and plots are in `tcontext/reports/` and `tcontext/artifacts/`.

---

## 🔁 Reproducibility

- Use `--seed 777` for deterministic sampling and train/val splits.
- The cleaned dataset cache (`tcontext/data/clean/`) is reused across runs unless manually deleted.
- The vector DB persists across sessions under `tcontext/vector_db/`.
- If no local TensorFlow Datasets zip is available, run a one-time TFDS download for `cats_vs_dogs` first.
