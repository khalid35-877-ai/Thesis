from __future__ import annotations

import csv
import json
import pickle
import random
import time
import uuid
from collections import Counter
from io import BytesIO
from pathlib import Path
import re
from urllib.request import Request, urlopen
import subprocess
import os as _os

# Suppress noisy framework logs before any heavy imports
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    chromadb = None

import numpy as np
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    LogisticRegression = None
    StandardScaler = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    pd = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    Image = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    tf = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    torch = None
    F = None

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - optional dependency for cloud deployments
    CLIPModel = None
    CLIPProcessor = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR
# Ensure all file operations are relative to the app's own directory,
# regardless of the working directory Streamlit was launched from.
import os as _os
_os.chdir(str(APP_DIR))

ARTIFACTS = PROJECT_DIR / "artifacts"
REPORTS = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "model" / "cats_vs_dogs_model_quick500.keras"
FALLBACK_MODEL_PATH = PROJECT_DIR / "model" / "cats_vs_dogs_model_quick500.pkl"
SUMMARY_PATH = ARTIFACTS / "sampled25_summary.json"
COMPARISON_CSV = ARTIFACTS / "sampled25_comparison_metrics.csv"
VECTOR_DB_ROOT = PROJECT_DIR / "vector_db"
FALLBACK_VECTOR_INDEX_PATH = VECTOR_DB_ROOT / "fallback_vector_index.json"
COLLECTION_NAME = "clip_image_embeddings"
CLASS_NAMES = ["cats", "dogs"]
CLIP_NAME = "openai/clip-vit-base-patch32"
COMPARATIVE_REPORT = REPORTS / "comparative_eda_report.md"
README_PATH = PROJECT_DIR / "README.md"
RUN_LOG_PATH = PROJECT_DIR.parent / "logs" / "run_sampled25_terminal.log"
MEMORY_MIN_CONFIDENCE = 0.75


def _find_sample_root() -> Path:
    """Return the best available sampled image directory."""
    for candidate in [
        PROJECT_DIR / "data" / "sampled_25_percent",
        PROJECT_DIR / "data" / "sampled_25_percent_seed_777",
        PROJECT_DIR / "data" / "sampled_10_percent",
    ]:
        if candidate.exists() and (candidate / "cats").exists() and (candidate / "dogs").exists():
            return candidate
    return PROJECT_DIR / "data" / "sampled_25_percent"


def _extract_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def _count_sampled_split():
    sample_root = _find_sample_root()
    cats = len(list((sample_root / "cats").glob("*.jpg"))) if (sample_root / "cats").exists() else 0
    dogs = len(list((sample_root / "dogs").glob("*.jpg"))) if (sample_root / "dogs").exists() else 0
    total = cats + dogs
    val = int(total * 0.2) if total else 0
    train = total - val if total else 0
    return train, val


def _extract_timing_from_log():
    if not RUN_LOG_PATH.exists():
        return {}
    raw = RUN_LOG_PATH.read_bytes()
    text = raw.decode("utf-16le", errors="ignore") if b"\x00" in raw[:200] else raw.decode("utf-8", errors="ignore")
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    text = ansi_re.sub("", text)

    # Keras epoch-end lines include total epoch duration like: "32/32 ... 23s ... - val_accuracy ..."
    epoch_secs = [int(v) for v in re.findall(r"32/32.*?\s([0-9]+)s.*?val_accuracy", text)]
    approx_train_seconds = float(sum(epoch_secs)) if epoch_secs else None
    return {"train_seconds": approx_train_seconds}


@st.cache_data(show_spinner=False)
def _read_json(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))


@st.cache_data(show_spinner=False)
def _read_text(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def _as_table(records, columns=None):
    if pd is not None:
        return pd.DataFrame(records, columns=columns)
    return list(records)


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    if pd is not None:
        return pd.read_csv(p)
    with p.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(show_spinner=False)
def _list_sample_images(sample_root_str: str):
    sample_root = Path(sample_root_str)
    cats = sorted((sample_root / "cats").glob("*.jpg")) if (sample_root / "cats").exists() else []
    dogs = sorted((sample_root / "dogs").glob("*.jpg")) if (sample_root / "dogs").exists() else []
    return [str(p) for p in cats], [str(p) for p in dogs]


def _image_feature_vector(img_or_path):
    if Image is None:
        return None
    if isinstance(img_or_path, (str, Path)):
        with Image.open(img_or_path) as img:
            rgb = np.array(img.convert("RGB"), dtype=np.float32)
    else:
        rgb = np.array(img_or_path.convert("RGB"), dtype=np.float32)
    gray = np.mean(rgb, axis=2)
    resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((16, 16)), dtype=np.float32)
    flattened = resized.reshape(-1) / 255.0
    color_stats = np.array(
        [rgb.mean(), rgb.std(), gray.mean(), gray.std(), np.mean(rgb[:, :, 0]), np.mean(rgb[:, :, 1]), np.mean(rgb[:, :, 2])],
        dtype=np.float32,
    )
    return np.concatenate([flattened, color_stats])


def _train_fallback_classifier():
    if Image is None or LogisticRegression is None or StandardScaler is None:
        return None
    sample_root = _find_sample_root()
    cats = sorted((sample_root / "cats").glob("*.jpg"))[:250]
    dogs = sorted((sample_root / "dogs").glob("*.jpg"))[:250]
    if not cats or not dogs:
        return None
    paths = cats + dogs
    labels = [0] * len(cats) + [1] * len(dogs)
    features = np.vstack([_image_feature_vector(p) for p in paths])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaled, labels)
    FALLBACK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FALLBACK_MODEL_PATH.open("wb") as handle:
        pickle.dump({"model": model, "scaler": scaler}, handle)
    return model, scaler


def _load_fallback_classifier():
    if FALLBACK_MODEL_PATH.exists():
        with FALLBACK_MODEL_PATH.open("rb") as handle:
            payload = pickle.load(handle)
        return payload.get("model"), payload.get("scaler")
    return _train_fallback_classifier()


class FallbackVectorCollection:
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = self._load_entries()

    def _load_entries(self):
        if not self.index_path.exists():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def _save_entries(self):
        self.index_path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")

    def count(self):
        return len(self._entries)

    def add(self, ids, embeddings, metadatas):
        for entry_id, embedding, metadata in zip(ids, embeddings, metadatas):
            self._entries.append(
                {"id": entry_id, "embedding": [float(x) for x in embedding], "metadata": metadata}
            )
        self._save_entries()

    def query(self, query_embeddings, n_results=5, include=None):
        if not self._entries:
            return {"metadatas": [[]], "distances": [[]]}
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        scored = []
        for entry in self._entries:
            embedding = np.asarray(entry["embedding"], dtype=np.float32)
            denom = np.linalg.norm(query) * np.linalg.norm(embedding)
            similarity = float(np.dot(query, embedding) / denom) if denom else 0.0
            distance = max(0.0, 1.0 - similarity)
            scored.append((distance, entry))
        scored.sort(key=lambda item: item[0])
        selected = scored[: max(1, int(n_results))]
        metadatas = [entry["metadata"] for _, entry in selected]
        distances = [distance for distance, _ in selected]
        return {"metadatas": [metadatas], "distances": [distances]}


@st.cache_resource
def load_classifier():
    if tf is not None and MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)
    fallback = _load_fallback_classifier()
    if fallback is None:
        return None
    return fallback


@st.cache_resource
def load_clip_stack():
    if torch is None or F is None or CLIPModel is None or CLIPProcessor is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    model.eval()
    return model, processor, device


@st.cache_resource
def load_vector_collection():
    if chromadb is not None and VECTOR_DB_ROOT.exists():
        db_candidates = [p for p in VECTOR_DB_ROOT.glob("clip_chroma_db*") if p.is_dir()]
        if db_candidates:
            best_path = None
            best_count = -1
            for p in db_candidates:
                try:
                    client = chromadb.PersistentClient(path=str(p))
                    col = client.get_or_create_collection(
                        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                    )
                    n = col.count()
                    if n > best_count:
                        best_count = n
                        best_path = p
                except Exception:
                    continue

            if best_path is None:
                best_path = sorted(db_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

            client = chromadb.PersistentClient(path=str(best_path))
            return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if FALLBACK_VECTOR_INDEX_PATH.exists() or Image is not None:
        collection = FallbackVectorCollection(FALLBACK_VECTOR_INDEX_PATH)
        if collection.count() == 0:
            sample_root = _find_sample_root()
            cats = sorted((sample_root / "cats").glob("*.jpg"))[:120]
            dogs = sorted((sample_root / "dogs").glob("*.jpg"))[:120]
            paths = cats + dogs
            if paths:
                embeddings = np.vstack([_image_feature_vector(p) for p in paths])
                ids = [f"fallback_{i:06d}" for i in range(len(paths))]
                metadatas = [
                    {"label": 0, "class_name": "cats", "path": str(p)} if p in cats else {"label": 1, "class_name": "dogs", "path": str(p)}
                    for p in paths
                ]
                collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
        return collection

    return None


def _encode_pil_images(images):
    stack = load_clip_stack()
    if stack is None:
        raise RuntimeError("CLIP model dependencies are not available in this deployment.")
    clip_model, processor, device = stack
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        emb_out = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = emb_out.pooler_output if hasattr(emb_out, "pooler_output") else emb_out
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def _build_memory_from_sampled_subset(max_per_class=200):
    if Image is None:
        return {"added": 0, "seconds": 0.0, "error": "Pillow is not available in this deployment."}

    sample_root = _find_sample_root()
    cats = sorted((sample_root / "cats").glob("*.jpg"))[:max_per_class]
    dogs = sorted((sample_root / "dogs").glob("*.jpg"))[:max_per_class]
    paths = [(p, 0, "cats") for p in cats] + [(p, 1, "dogs") for p in dogs]
    if not paths:
        return {"added": 0, "seconds": 0.0}

    t0 = time.time()
    if chromadb is not None:
        bootstrap_db = VECTOR_DB_ROOT / "clip_chroma_db_manual"
        bootstrap_db.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(bootstrap_db))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

        batch_size = 32
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            imgs = [Image.open(p).convert("RGB") for p, _, _ in batch]
            try:
                emb = _encode_pil_images(imgs)
            except RuntimeError:
                return {"added": 0, "seconds": time.time() - t0, "error": "CLIP model dependencies are not available in this deployment."}
            ids = [f"bootstrap_{i+start:06d}" for i in range(len(batch))]
            metas = [{"label": int(lbl), "class_name": cname, "path": str(p)} for (p, lbl, cname) in batch]
            collection.add(ids=ids, embeddings=emb.tolist(), metadatas=metas)
            for img in imgs:
                img.close()
        return {"added": len(paths), "seconds": time.time() - t0}

    collection = FallbackVectorCollection(FALLBACK_VECTOR_INDEX_PATH)
    embeddings = []
    for path, _, _ in paths:
        feature = _image_feature_vector(path)
        if feature is not None:
            embeddings.append(feature)
    if not embeddings:
        return {"added": 0, "seconds": time.time() - t0, "error": "No image features could be generated."}
    collection.add(
        ids=[f"fallback_bootstrap_{i:06d}" for i in range(len(paths))],
        embeddings=embeddings,
        metadatas=[{"label": int(lbl), "class_name": cname, "path": str(p)} for (p, lbl, cname) in paths],
    )
    return {"added": len(paths), "seconds": time.time() - t0}


def _load_image_from_url(url: str):
    if Image is None:
        raise RuntimeError("Pillow is not available in this deployment.")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=12) as resp:
        payload = resp.read()
    return Image.open(BytesIO(payload)).convert("RGB")


def classify_image(img: Image.Image):
    if Image is None:
        return None
    model = load_classifier()
    if model is None:
        return None
    if tf is not None and isinstance(model, tf.keras.Model):
        x = img.convert("RGB").resize((224, 224))
        arr = np.array(x, dtype=np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        prob_dog = float(model.predict(arr, verbose=0).reshape(-1)[0])
        pred = "dogs" if prob_dog > 0.5 else "cats"
        conf = prob_dog if pred == "dogs" else 1.0 - prob_dog
        return {"prediction": pred, "confidence": conf, "dog_probability": prob_dog}
    if isinstance(model, tuple):
        classifier, scaler = model
        features = _image_feature_vector(img)
        if features is None or classifier is None or scaler is None:
            return None
        prob_dog = float(classifier.predict_proba(scaler.transform([features]))[0, 1])
        pred = "dogs" if prob_dog > 0.5 else "cats"
        conf = prob_dog if pred == "dogs" else 1.0 - prob_dog
        return {"prediction": pred, "confidence": conf, "dog_probability": prob_dog}
    return None


def retrieval_predict(img: Image.Image, top_k=5):
    if Image is None:
        return None
    collection = load_vector_collection()
    if collection is None or collection.count() == 0:
        return None
    stack = load_clip_stack()
    if stack is None:
        return None
    clip_model, processor, device = stack
    inputs = processor(images=[img.convert("RGB")], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        emb_out = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = emb_out.pooler_output if hasattr(emb_out, "pooler_output") else emb_out
        emb = F.normalize(emb, dim=-1)
    result = collection.query(query_embeddings=emb.cpu().numpy().tolist(), n_results=top_k, include=["metadatas", "distances"])
    neighbors = result["metadatas"][0]
    distances = result["distances"][0]
    label_votes = [int(m["label"]) for m in neighbors]
    pred_idx = Counter(label_votes).most_common(1)[0][0]
    pred_name = CLASS_NAMES[pred_idx]
    nn_table = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "class": neighbors[i]["class_name"],
                "distance": float(distances[i]),
            }
            for i in range(len(neighbors))
        ]
    )
    return {"prediction": pred_name, "neighbors": nn_table}


def add_memory_image(img: Image.Image, label: str, source_name: str):
    if Image is None:
        raise RuntimeError("Image processing is not available in this deployment.")
    collection = load_vector_collection()
    if collection is None:
        VECTOR_DB_ROOT.mkdir(parents=True, exist_ok=True)
        collection = load_vector_collection.clear() or load_vector_collection()
    stack = load_clip_stack()
    if stack is None:
        raise RuntimeError("CLIP model dependencies are not available in this deployment.")
    clip_model, processor, device = stack
    inputs = processor(images=[img.convert("RGB")], return_tensors="pt", padding=True).to(device)
    t0 = time.time()
    with torch.inference_mode():
        emb_out = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = emb_out.pooler_output if hasattr(emb_out, "pooler_output") else emb_out
        emb = F.normalize(emb, dim=-1)
    add_seconds = time.time() - t0
    label_idx = 0 if label == "cats" else 1
    doc_id = f"web_{label}_{uuid.uuid4().hex[:10]}"
    collection.add(
        ids=[doc_id],
        embeddings=emb.cpu().numpy().tolist(),
        metadatas=[{"label": label_idx, "class_name": label, "path": source_name}],
    )
    return {"id": doc_id, "add_seconds": add_seconds}


def _passes_memory_threshold(img: Image.Image, expected_label: str, min_confidence: float = MEMORY_MIN_CONFIDENCE):
    # Self-cleaning gate: reject uncertain or label-mismatched images.
    clf = classify_image(img)
    if clf is None:
        return False, "Classifier is unavailable for threshold check."

    predicted_label = clf["prediction"]
    confidence = float(clf["confidence"])
    if predicted_label != expected_label:
        return (
            False,
            f"Label mismatch (predicted={predicted_label}, expected={expected_label}, confidence={confidence:.3f}).",
        )
    if confidence < min_confidence:
        return (
            False,
            f"Confidence {confidence:.3f} is below threshold {min_confidence:.2f}.",
        )

    return True, f"Accepted (confidence={confidence:.3f})."


def render_metrics():
    summary = None
    if SUMMARY_PATH.exists():
        summary = _read_json(str(SUMMARY_PATH))
    else:
        # Fallback for presentation when artifacts were cleaned but README/report exist.
        if README_PATH.exists():
            readme_text = _read_text(str(README_PATH))
            acc_clf = re.search(r"Classifier metrics[\s\S]*?- Accuracy: `([0-9.]+)`", readme_text)
            pre_clf = re.search(r"Classifier metrics[\s\S]*?- Precision: `([0-9.]+)`", readme_text)
            rec_clf = re.search(r"Classifier metrics[\s\S]*?- Recall: `([0-9.]+)`", readme_text)
            f1_clf = re.search(r"Classifier metrics[\s\S]*?- F1: `([0-9.]+)`", readme_text)
            acc_ret = re.search(r"Retrieval metrics[\s\S]*?- Accuracy: `([0-9.]+)`", readme_text)
            pre_ret = re.search(r"Retrieval metrics[\s\S]*?- Precision: `([0-9.]+)`", readme_text)
            rec_ret = re.search(r"Retrieval metrics[\s\S]*?- Recall: `([0-9.]+)`", readme_text)
            f1_ret = re.search(r"Retrieval metrics[\s\S]*?- F1: `([0-9.]+)`", readme_text)
            if all([acc_clf, pre_clf, rec_clf, f1_clf, acc_ret, pre_ret, rec_ret, f1_ret]):
                train_sec = _extract_float(r"Training time \(s\): ([0-9.]+)", readme_text)
                infer_clf_sec = _extract_float(r"Inference time total \(s\): ([0-9.]+)", readme_text)
                infer_ret_sec = _extract_float(r"Retrieval inference total \(s\): ([0-9.]+)", readme_text)
                db_build_sec = _extract_float(r"Retrieval DB build time \(s\): ([0-9.]+)", readme_text)
                train_samples, val_samples = _count_sampled_split()
                summary = {
                    "classifier": {
                        "accuracy": float(acc_clf.group(1)),
                        "precision": float(pre_clf.group(1)),
                        "recall": float(rec_clf.group(1)),
                        "f1": float(f1_clf.group(1)),
                        "train_seconds": train_sec,
                        "inference_seconds": infer_clf_sec,
                    },
                    "retrieval": {
                        "accuracy": float(acc_ret.group(1)),
                        "precision": float(pre_ret.group(1)),
                        "recall": float(rec_ret.group(1)),
                        "f1": float(f1_ret.group(1)),
                        "inference_seconds": infer_ret_sec,
                        "db_build_seconds": db_build_sec,
                    },
                    "dataset": {"train_samples": train_samples, "test_samples": val_samples},
                    "_fallback": True,
                }
                if train_sec is None:
                    log_timing = _extract_timing_from_log()
                    if "train_seconds" in log_timing:
                        summary["classifier"]["train_seconds"] = log_timing["train_seconds"]

    if summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Classifier Acc", f"{summary['classifier']['accuracy']:.4f}")
        col2.metric("Retrieval Acc", f"{summary['retrieval']['accuracy']:.4f}")
        col3.metric("Train Samples", f"{summary['dataset']['train_samples']}")
        col4.metric("Validation Samples", f"{summary['dataset']['test_samples']}")
        if summary.get("_fallback"):
            st.caption("Using fallback metrics from `tcontext/README.md` snapshot.")
        if "train_seconds" in summary.get("classifier", {}) or "db_build_seconds" in summary.get("retrieval", {}):
            st.markdown("#### Time and Computational Profile")
            t1, t2, t3, t4 = st.columns(4)
            dl_train = summary["classifier"].get("train_seconds")
            dl_inf = summary["classifier"].get("inference_seconds")
            ret_build = summary["retrieval"].get("db_build_seconds")
            ret_inf = summary["retrieval"].get("inference_seconds")
            t1.metric("DL Train Time", f"{dl_train:.2f} s" if isinstance(dl_train, (int, float)) else "N/A")
            t2.metric("DL Inference Time", f"{dl_inf:.4f} s" if isinstance(dl_inf, (int, float)) else "N/A")
            t3.metric("Retrieval Build Time", f"{ret_build:.2f} s" if isinstance(ret_build, (int, float)) else "N/A")
            t4.metric("Retrieval Inference", f"{ret_inf:.4f} s" if isinstance(ret_inf, (int, float)) else "N/A")
            if summary.get("_fallback") and not all(isinstance(v, (int, float)) for v in [dl_train, dl_inf, ret_build, ret_inf]):
                st.caption("Exact timing metrics require `sampled25_summary.json`. Use 'Rebuild Full Experiment Artifacts' for full values.")

        # Single comparative table for thesis-ready side-by-side reading.
        st.markdown("#### Unified Comparative View")
        cmp_df = _as_table(
            [
                {
                    "method": "Deep Learning (EfficientNet Classifier)",
                    "accuracy": float(summary["classifier"]["accuracy"]),
                    "precision": float(summary["classifier"]["precision"]),
                    "recall": float(summary["classifier"]["recall"]),
                    "f1": float(summary["classifier"]["f1"]),
                    "train_or_build_sec": summary["classifier"].get("train_seconds"),
                    "inference_sec": summary["classifier"].get("inference_seconds"),
                },
                {
                    "method": "Retrieval (CLIP + Vector DB)",
                    "accuracy": float(summary["retrieval"]["accuracy"]),
                    "precision": float(summary["retrieval"]["precision"]),
                    "recall": float(summary["retrieval"]["recall"]),
                    "f1": float(summary["retrieval"]["f1"]),
                    "train_or_build_sec": summary["retrieval"].get("db_build_seconds"),
                    "inference_sec": summary["retrieval"].get("inference_seconds"),
                },
            ],
            columns=[
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "train_or_build_sec",
                "inference_sec",
            ],
        )
        st.dataframe(cmp_df, width="stretch", hide_index=True)

        acc_delta = float(summary["classifier"]["accuracy"] - summary["retrieval"]["accuracy"])
        f1_delta = float(summary["classifier"]["f1"] - summary["retrieval"]["f1"])
        st.caption(
            f"Delta (Classifier - Retrieval): accuracy={acc_delta:+.4f}, f1={f1_delta:+.4f}."
        )
        if "incremental_cost_estimates" in summary:
            st.markdown("#### Incremental Update Cost (Estimated)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("DL Refit (5 imgs)", f"{summary['incremental_cost_estimates']['deep_learning_refit_5_images_seconds']:.2f} s")
            c2.metric("DL Refit (10 imgs)", f"{summary['incremental_cost_estimates']['deep_learning_refit_10_images_seconds']:.2f} s")
            c3.metric("Retrieval Add (5)", f"{summary['incremental_cost_estimates']['retrieval_add_5_images_seconds']:.2f} s")
            c4.metric("Retrieval Add (10)", f"{summary['incremental_cost_estimates']['retrieval_add_10_images_seconds']:.2f} s")
        if "compute_environment" in summary:
            st.caption(
                "Compute profile - "
                f"TF GPU: {summary['compute_environment']['tensorflow_gpu_available']}, "
                f"Torch CUDA: {summary['compute_environment']['pytorch_cuda_available']}, "
                f"CLIP device: {summary['compute_environment']['clip_device_used']}"
            )
    else:
        st.warning("No summary file found. Run the experiment first.")


def render_project_intent():
    st.subheader("Project Intention and Thesis Claim")
    st.info(
        "Goal: compare two separate systems for the same cat-vs-dog task - "
        "(1) deep learning classifier (retraining-based) and "
        "(2) retrieval memory system (CLIP + vector DB, no retraining for updates)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Deep Learning Path")
        st.markdown("- Train model weights with epochs")
        st.markdown("- Higher training compute/time cost")
        st.markdown("- Strong final predictive performance")
        st.markdown("- Incremental updates require fine-tuning/retraining")
    with c2:
        st.markdown("### Retrieval Path")
        st.markdown("- Encode images and store vectors in memory")
        st.markdown("- Lower update-time computational cost")
        st.markdown("- Very fast incremental memory insertion")
        st.markdown("- Competitive performance with high operational agility")


def render_experiment_status():
    st.subheader("Experiment Status")
    has_fallback_model = FALLBACK_MODEL_PATH.exists() or (load_classifier() is not None)
    has_vector_assets = (VECTOR_DB_ROOT.exists() and (any(VECTOR_DB_ROOT.glob("clip_chroma_db*")) or FALLBACK_VECTOR_INDEX_PATH.exists())) or (load_vector_collection() is not None)
    required = {
        "Summary JSON (optional)": SUMMARY_PATH.exists(),
        "Comparison CSV (optional)": COMPARISON_CSV.exists(),
        "Classifier model": MODEL_PATH.exists() or has_fallback_model,
        "Vector DB": has_vector_assets,
        "Comparative EDA report": COMPARATIVE_REPORT.exists(),
    }
    cols = st.columns(len(required))
    for i, (name, ok) in enumerate(required.items()):
        cols[i].metric(name, "Ready" if ok else "Missing")
    if not (MODEL_PATH.exists() or has_fallback_model) or not has_vector_assets:
        st.warning("Core demo assets are missing. The app will build lightweight fallback assets on demand.")
    elif not (SUMMARY_PATH.exists() and COMPARISON_CSV.exists()):
        st.info("Core demo is ready. Optional summary files are missing; app is using fallback sources.")

    collection = load_vector_collection()
    db_count = 0 if collection is None else collection.count()
    st.caption(f"Indexed vectors available: {db_count}")
    if db_count == 0:
        st.warning("Retrieval memory is empty. Initialize once for fully functional demo.")

    st.markdown("#### Quick Actions")
    c1, c2 = st.columns(2)
    with c1:
        per_class = st.select_slider("Bootstrap images per class", options=[50, 100, 200, 400, 625], value=200)
        if st.button("Initialize Retrieval Memory"):
            with st.spinner("Indexing sampled images into vector DB..."):
                res = _build_memory_from_sampled_subset(max_per_class=per_class)
            load_vector_collection.clear()
            st.success(f"Indexed {res['added']} images in {res['seconds']:.2f}s.")
    with c2:
        if st.button("Generate Comparative EDA"):
            proc = subprocess.run(
                ["python", str(PROJECT_DIR / "comparative_eda.py")],
                capture_output=True, text=True, cwd=str(PROJECT_DIR)
            )
            if proc.returncode == 0:
                st.success("Comparative EDA report generated.")
            else:
                st.error("Could not generate comparative EDA report.")
                st.code(proc.stderr or proc.stdout)
    if st.button("Rebuild Full Experiment Artifacts (slow)"):
        with st.spinner("Running full pipeline to regenerate summary/csv/model artifacts..."):
            proc = subprocess.run(
                ["python", str(PROJECT_DIR / "quick500_experiment.py"), "--seed", "777"],
                capture_output=True, text=True, cwd=str(PROJECT_DIR)
            )
        if proc.returncode == 0:
            st.success("Full artifacts rebuilt. Refresh page to load complete metrics.")
        else:
            st.error("Full artifact rebuild failed.")
            st.code(proc.stderr or proc.stdout)


def render_artifacts():
    st.subheader("Experiment Artifacts")
    if COMPARISON_CSV.exists():
        comp_df = _read_csv(str(COMPARISON_CSV))
        if comp_df is not None:
            st.dataframe(comp_df, width="stretch")
    for img_name in [
        "eda_class_distribution.png",
        "eda_size_distribution.png",
        "training_curves.png",
        "classifier_roc_curve.png",
        "comparison_confusion_matrices.png",
    ]:
        img_path = ARTIFACTS / img_name
        if img_path.exists():
            st.image(str(img_path), caption=img_name)


def render_comparative_eda():
    st.subheader("Comparative EDA (Deep Learning vs Retrieval)")
    cmp_plot = ARTIFACTS / "comparative_method_metrics.png"
    curve_plot = ARTIFACTS / "comparative_learning_curve.png"
    need_generate = (not cmp_plot.exists()) or (not curve_plot.exists()) or (not COMPARATIVE_REPORT.exists())
    if need_generate:
        with st.spinner("Preparing comparative EDA assets..."):
            proc = subprocess.run(
                ["python", str(PROJECT_DIR / "comparative_eda.py")],
                capture_output=True, text=True, cwd=str(PROJECT_DIR)
            )
        if proc.returncode != 0:
            st.error("Could not generate comparative EDA assets.")
            st.code(proc.stderr or proc.stdout)
        else:
            _read_text.clear()

    if cmp_plot.exists():
        st.image(str(cmp_plot), caption="Comparative metrics: classifier vs retrieval")
    if curve_plot.exists():
        st.image(str(curve_plot), caption="Deep learning convergence behavior")
    if COMPARATIVE_REPORT.exists():
        report_text = _read_text(str(COMPARATIVE_REPORT))
        if report_text:
            st.markdown(report_text)
    elif RUN_LOG_PATH.exists():
        st.info("Comparative report missing, showing latest run-log based evidence.")
        st.code("Run log found at logs/run_sampled25_terminal.log. Generate full report with: python tcontext/comparative_eda.py")
    else:
        st.warning("Comparative EDA report not found. Run: `python tcontext/comparative_eda.py`")


def render_live_demo():
    st.subheader("Live Comparative Inference")
    if Image is None:
        st.warning("Image processing is unavailable because Pillow is not installed in this deployment.")
        return
    st.markdown("Choose an image source:")
    source_mode = st.radio(
        "Image source",
        options=["Built-in sample", "Upload", "External file path", "Image URL"],
        horizontal=True,
    )

    img = None
    sample_root = _find_sample_root()
    if source_mode == "Upload":
        uploaded = st.file_uploader("Upload a cat/dog image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Input Image", width=320)
    elif source_mode == "External file path":
        ext_path = st.text_input("Enter external/local image file path", value="")
        if ext_path.strip():
            p = Path(ext_path.strip())
            if p.exists():
                img = Image.open(p).convert("RGB")
                st.image(img, caption=f"External file: {p.name}", width=320)
            else:
                st.error("Path not found. Please provide a valid image path.")
    elif source_mode == "Image URL":
        image_url = st.text_input("Enter image URL", value="")
        if image_url.strip():
            try:
                img = _load_image_from_url(image_url.strip())
                st.image(img, caption="Image from URL", width=320)
            except Exception as ex:
                st.error(f"Could not load image from URL: {ex}")
    else:
        cats, dogs = _list_sample_images(str(sample_root))
        samples = cats + dogs
        if not samples:
            st.warning("No local sample images found. Run experiment once, or switch to upload mode.")
            return
        picked = st.selectbox("Select a sample image", options=[str(p) for p in samples], index=0)
        cbtn1, cbtn2 = st.columns(2)
        with cbtn1:
            if st.button("Pick random CAT sample"):
                if cats:
                    picked = str(random.choice(cats))
                st.session_state["picked_sample"] = picked
        with cbtn2:
            if st.button("Pick random DOG sample"):
                if dogs:
                    picked = str(random.choice(dogs))
                st.session_state["picked_sample"] = picked
        if "picked_sample" in st.session_state:
            picked = st.session_state["picked_sample"]
        img = Image.open(picked).convert("RGB")
        st.image(img, caption=f"Built-in Sample: {Path(picked).name}", width=320)

    if img is None:
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### EfficientNet Classifier")
        clf = classify_image(img)
        if clf is None:
            st.error("Classifier model not found. Run training first.")
        else:
            st.success(f"Prediction: **{clf['prediction']}**")
            st.write(f"Confidence: `{clf['confidence']:.4f}`")
            st.write(f"P(dog): `{clf['dog_probability']:.4f}`")

    with c2:
        st.markdown("#### CLIP + Vector DB Retrieval")
        ret = retrieval_predict(img, top_k=5)
        if ret is None:
            st.error("Vector DB not found or empty. Use 'Initialize Retrieval Memory' in Overview.")
        else:
            st.success(f"Prediction: **{ret['prediction']}**")
            st.dataframe(ret["neighbors"], width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### Incremental Retrieval Memory (No Retraining)")
    memory_uploads = st.file_uploader(
        "Add 1-10 new memory images to retrieval DB",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="memory_uploader",
    )
    memory_label = st.selectbox("Label for uploaded memory images", options=CLASS_NAMES, index=0)
    if st.button("Add to Retrieval Memory"):
        if not memory_uploads:
            st.warning("Upload at least one image.")
            return
        if len(memory_uploads) > 10:
            st.error("Please upload at most 10 images per update.")
            return
        add_times = []
        accepted = 0
        rejected = 0
        rejected_details = []
        for up in memory_uploads:
            memory_img = Image.open(BytesIO(up.read())).convert("RGB")
            keep, reason = _passes_memory_threshold(memory_img, memory_label)
            if not keep:
                rejected += 1
                rejected_details.append(f"{up.name}: {reason}")
                continue
            res = add_memory_image(memory_img, memory_label, up.name)
            add_times.append(res["add_seconds"])
            accepted += 1

        if accepted > 0:
            st.success(
                f"Added {accepted} images into retrieval memory. "
                f"Total add time: {sum(add_times):.3f}s, Avg/image: {np.mean(add_times):.3f}s"
            )
            st.info(f"Threshold check: accepted images met the threshold of {MEMORY_MIN_CONFIDENCE * 100:.0f}%.")
        else:
            st.warning(
                f"No image met the threshold of {MEMORY_MIN_CONFIDENCE * 100:.0f}%, so nothing was added."
            )

        if rejected_details:
            st.markdown(f"**Rejected for not meeting threshold ({MEMORY_MIN_CONFIDENCE * 100:.0f}%):**")
            for line in rejected_details:
                st.write(f"- {line}")

        st.info("Retrieval system is updated instantly. Deep learning model would require re-training/fine-tuning for new memory.")


def _run_command_live(cmd: list[str], cwd: str):
    """Stream a subprocess command's output line-by-line into a Streamlit code block."""
    import sys
    output_area = st.empty()
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
        for line in proc.stdout:
            lines.append(line.rstrip())
            # Keep last 60 lines visible so the box doesn't grow forever
            output_area.code("\n".join(lines[-60:]), language="text")
        proc.wait()
        return proc.returncode
    except FileNotFoundError as exc:
        st.error(f"Could not start process: {exc}")
        return 1


def render_commands():
    st.subheader("Run Commands")
    st.caption("Every command runs from the project root. Click the button to execute and stream live output here.")

    REPO_ROOT = str(PROJECT_DIR.parent)
    TCONTEXT = str(PROJECT_DIR)

    # ------------------------------------------------------------------ #
    # Command definitions: (label, description, cmd_list, cwd, key)       #
    # ------------------------------------------------------------------ #
    commands = [
        {
            "label": "Launch Streamlit App",
            "description": "Start the dashboard on `http://localhost:8501`. "
                           "Use this if you launched the app from the terminal and want a reminder of the command.",
            "cmd_display": "python -m streamlit run main.py",
            "cmd": ["python", "-m", "streamlit", "run", str(PROJECT_DIR.parent / "main.py")],
            "cwd": REPO_ROOT,
            "key": "cmd_launch",
            "warn": "This will open a second Streamlit instance — you are already inside one. "
                    "Use it only as a reference or run it from your terminal instead.",
            "readonly": True,
        },
        {
            "label": "Install Dependencies",
            "description": "Install all pinned packages from `tcontext/requirements.txt`.",
            "cmd_display": "pip install -r tcontext\\requirements.txt",
            "cmd": ["pip", "install", "-r", str(PROJECT_DIR / "requirements.txt")],
            "cwd": REPO_ROOT,
            "key": "cmd_install",
        },
        {
            "label": "Run Full Experiment Pipeline",
            "description": "Train EfficientNetB0, build CLIP vector index, evaluate both methods, "
                           "save all artifacts. Takes several minutes on first run.",
            "cmd_display": "python tcontext\\quick500_experiment.py --seed 777",
            "cmd": ["python", str(PROJECT_DIR / "quick500_experiment.py"), "--seed", "777"],
            "cwd": REPO_ROOT,
            "key": "cmd_experiment",
        },
        {
            "label": "Generate Comparative EDA",
            "description": "Produce comparative method metrics plot, learning curve plot, "
                           "and `reports/comparative_eda_report.md`.",
            "cmd_display": "python tcontext\\comparative_eda.py",
            "cmd": ["python", str(PROJECT_DIR / "comparative_eda.py")],
            "cwd": REPO_ROOT,
            "key": "cmd_eda",
        },
        {
            "label": "Query Image — Vector DB Retrieval",
            "description": "Run a retrieval prediction for a single image against the vector DB.",
            "cmd_display": 'python tcontext\\query_demo.py --query-image "path\\to\\image.jpg" --top-k 5',
            "cmd": None,   # path is user-provided — shown as copy-only
            "cwd": REPO_ROOT,
            "key": "cmd_query",
        },
        {
            "label": "Add Image to Retrieval Memory",
            "description": "Instantly index a new labeled image into the vector DB without retraining.",
            "cmd_display": 'python tcontext\\query_demo.py --add-image "path\\to\\image.jpg" --label dogs',
            "cmd": None,   # path is user-provided — shown as copy-only
            "cwd": REPO_ROOT,
            "key": "cmd_add",
        },
        {
            "label": "Capture Epoch Logs to File",
            "description": "Run the full experiment and save all output (including Keras epoch logs) to a log file.",
            "cmd_display": "python tcontext\\quick500_experiment.py --seed 777 2>&1 | Tee-Object -FilePath logs\\run_sampled25_terminal.log",
            "cmd": None,   # PowerShell pipeline — copy-only
            "cwd": REPO_ROOT,
            "key": "cmd_log",
        },
    ]

    for entry in commands:
        with st.expander(f"**{entry['label']}**  —  {entry['description']}", expanded=False):
            st.code(entry["cmd_display"], language="powershell")

            if entry.get("readonly"):
                st.info(entry.get("warn", ""))
                continue

            if entry["cmd"] is None:
                st.caption("This command requires a file path — copy it above, fill in your path, and run it in your terminal.")
                continue

            run_key = f"run_{entry['key']}"
            if st.button(f"▶ Run: {entry['label']}", key=run_key):
                st.markdown("**Output:**")
                rc = _run_command_live(entry["cmd"], cwd=entry["cwd"])
                if rc == 0:
                    st.success("Command completed successfully.")
                    # Clear caches so the dashboard reflects new artifacts immediately
                    st.cache_data.clear()
                    st.cache_resource.clear()
                else:
                    st.error(f"Command exited with code {rc}.")

    # ------------------------------------------------------------------ #
    # Quick reference card                                                 #
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown("#### Quick Reference")
    ref = {
        "One-click launch (PowerShell)": ".\\launch_app.ps1",
        "Full pipeline + launch": ".\\run_demo.ps1",
        "Install deps": "pip install -r tcontext\\requirements.txt",
        "Train + benchmark": "python tcontext\\quick500_experiment.py --seed 777",
        "EDA report": "python tcontext\\comparative_eda.py",
        "Query vector DB": 'python tcontext\\query_demo.py --query-image "img.jpg" --top-k 5',
        "Add to memory": 'python tcontext\\query_demo.py --add-image "img.jpg" --label dogs',
    }
    ref_df = _as_table(list(ref.items()), columns=["Action", "Command"])
    st.dataframe(ref_df, width="stretch", hide_index=True)


def main():
    try:
        st.set_page_config(page_title="Thesis Comparative Demo", layout="wide")
    except Exception:
        pass
    st.title("Thesis Demo: Deep Learning vs Retrieval Memory")
    st.caption("Clear objective: show similar task performance, but much lower incremental update cost for retrieval-based memory")
    section = st.radio(
        "Section",
        options=["Overview", "Comparative EDA", "Artifacts", "Live Demo", "Commands"],
        horizontal=True,
    )

    if section == "Overview":
        render_project_intent()
        render_metrics()
        render_experiment_status()
    elif section == "Comparative EDA":
        render_comparative_eda()
    elif section == "Artifacts":
        render_artifacts()
    elif section == "Commands":
        render_commands()
    else:
        render_live_demo()


if __name__ == "__main__":
    main()
