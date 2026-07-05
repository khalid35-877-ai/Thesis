import argparse
import json
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import layers
from transformers import CLIPModel, CLIPProcessor

from dataset_utils import get_dataset_root


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SAMPLE_RATIO = 0.25
EPOCHS = 12
PROJECT_DIR = Path(__file__).resolve().parent
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_quick_subset(seed: int, sample_ratio: float = SAMPLE_RATIO) -> Path:
    source_root = get_dataset_root()
    out_root = PROJECT_DIR / "data" / f"sampled_25_percent_seed_{seed}"
    cats_dst = out_root / "cats"
    dogs_dst = out_root / "dogs"

    if out_root.exists():
        shutil.rmtree(out_root)
    cats_dst.mkdir(parents=True, exist_ok=True)
    dogs_dst.mkdir(parents=True, exist_ok=True)

    cats_all = sorted((source_root / "cats").glob("*.jpg"))
    dogs_all = sorted((source_root / "dogs").glob("*.jpg"))

    cats_k = max(1, int(len(cats_all) * sample_ratio))
    dogs_k = max(1, int(len(dogs_all) * sample_ratio))
    rng = random.Random(seed)
    cats_src = rng.sample(cats_all, cats_k)
    dogs_src = rng.sample(dogs_all, dogs_k)

    for i, f in enumerate(cats_src):
        shutil.copy2(f, cats_dst / f"{i:04d}.jpg")
    for i, f in enumerate(dogs_src):
        shutil.copy2(f, dogs_dst / f"{i:04d}.jpg")

    return out_root


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def split_dataset_files(data_root: Path, seed: int, val_ratio: float = 0.2):
    rng = random.Random(seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_names = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    for class_idx, class_name in enumerate(class_names):
        files = sorted((data_root / class_name).glob("*.jpg"))
        rng.shuffle(files)
        val_count = int(len(files) * val_ratio)
        class_val = files[:val_count]
        class_train = files[val_count:]

        train_paths.extend(class_train)
        train_labels.extend([class_idx] * len(class_train))
        val_paths.extend(class_val)
        val_labels.extend([class_idx] * len(class_val))

    return (train_paths, np.array(train_labels)), (val_paths, np.array(val_labels))


def load_resized_images(paths):
    def _load_one(p):
        with Image.open(p) as img:
            return np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.uint8)

    if not paths:
        return np.empty((0, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

    max_workers = min(8, max(1, len(paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(_load_one, paths))

    return np.stack(images, axis=0)


def make_tf_dataset(images, labels, training: bool, batch_size: int, seed: int, augmentation=None):
    image_tensor = tf.constant(images, dtype=tf.uint8)
    label_tensor = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((image_tensor, label_tensor))

    if training:
        ds = ds.shuffle(buffer_size=int(images.shape[0]), seed=seed, reshuffle_each_iteration=True)

    def _load_and_preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if training and augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def _clip_encode_images(paths, model, processor, device, batch_size=32):
    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=inputs["pixel_values"])
            if hasattr(features, "image_embeds"):
                features = features.image_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features, dim=-1)
        embeddings.append(features.cpu().numpy())

    return np.vstack(embeddings)


def get_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def _cleanup_output_dirs(artifacts: Path, reports: Path):
    # Keep previous outputs visible in Streamlit while a new run is executing.
    # Files generated in this run will overwrite existing outputs at the end.
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)


def _plot_eda(data_root: Path, class_names, artifacts: Path):
    counts = Counter()
    heights, widths = [], []
    samples = []

    for class_name in class_names:
        files = sorted((data_root / class_name).glob("*.jpg"))
        counts[class_name] = len(files)
        for i, fp in enumerate(files):
            with Image.open(fp) as img:
                widths.append(img.size[0])
                heights.append(img.size[1])
            if i < 8:
                samples.append((fp, class_name))

    plt.figure(figsize=(6, 4))
    plt.bar(list(counts.keys()), list(counts.values()), color=["#6baed6", "#fd8d3c"])
    plt.title("Class Distribution (25% Sample)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(artifacts / "eda_class_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(heights, bins=20, color="#6baed6")
    axes[0].set_title("Image Height Distribution")
    axes[1].hist(widths, bins=20, color="#fd8d3c")
    axes[1].set_title("Image Width Distribution")
    plt.tight_layout()
    plt.savefig(artifacts / "eda_size_distribution.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    for i, (fp, label) in enumerate(samples[:16]):
        plt.subplot(4, 4, i + 1)
        with Image.open(fp) as img:
            plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(artifacts / "eda_samples.png", dpi=180)
    plt.close(fig)

    eda_by_class = {}
    for class_name in class_names:
        class_files = sorted((data_root / class_name).glob("*.jpg"))
        class_h, class_w = [], []
        for fp in class_files:
            with Image.open(fp) as img:
                class_w.append(img.size[0])
                class_h.append(img.size[1])
        eda_by_class[class_name] = {
            "count": len(class_files),
            "height_mean": float(np.mean(class_h)) if class_h else 0.0,
            "height_std": float(np.std(class_h)) if class_h else 0.0,
            "width_mean": float(np.mean(class_w)) if class_w else 0.0,
            "width_std": float(np.std(class_w)) if class_w else 0.0,
        }

    return counts, heights, widths, eda_by_class


def run(seed: int, query_image: str | None = None):
    set_seed(seed)
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifacts = PROJECT_DIR / "artifacts"
    reports = PROJECT_DIR / "reports"
    model_dir = PROJECT_DIR / "model"
    _cleanup_output_dirs(artifacts, reports)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_root = prepare_quick_subset(seed=seed, sample_ratio=SAMPLE_RATIO)
    class_names = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    (train_paths, train_labels), (val_paths, val_labels) = split_dataset_files(data_root, seed=seed, val_ratio=0.2)
    train_images = load_resized_images(train_paths)
    val_images = load_resized_images(val_paths)

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
        ],
        name="augmentation",
    )
    train_ds = make_tf_dataset(
        train_images, train_labels, training=True, batch_size=BATCH_SIZE, seed=seed, augmentation=augmentation
    )
    val_ds = make_tf_dataset(val_images, val_labels, training=False, batch_size=BATCH_SIZE, seed=seed)

    counts, h, w, eda_by_class = _plot_eda(data_root, class_names, artifacts)

    # Classifier
    base = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    clf = tf.keras.Model(inputs=inputs, outputs=outputs)
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model_summary_buffer = StringIO()
    clf.summary(print_fn=lambda x: model_summary_buffer.write(x + "\n"))
    model_summary_text = model_summary_buffer.getvalue()
    total_params = int(clf.count_params())
    trainable_params = int(np.sum([np.prod(v.shape) for v in clf.trainable_weights]))

    t0 = time.time()
    history = clf.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=1, min_lr=1e-6),
        ],
        verbose=1,
    )
    train_seconds = time.time() - t0
    clf.save(model_dir / "cats_vs_dogs_model_quick500.keras")
    model_size_mb = float((model_dir / "cats_vs_dogs_model_quick500.keras").stat().st_size / (1024**2))

    y_true, y_prob = [], []
    t1 = time.time()
    for images, labels in val_ds:
        p = clf.predict(images, verbose=0).reshape(-1)
        y_prob.extend(p.tolist())
        y_true.extend(labels.numpy().tolist())
    clf_infer_seconds = time.time() - t1
    y_true = np.array(y_true).astype(int)
    y_pred_clf = (np.array(y_prob) > 0.5).astype(int)
    clf_metrics = get_metrics(y_true, y_pred_clf)
    clf_auc = float(roc_auc_score(y_true, np.array(y_prob)))
    clf_report = classification_report(y_true, y_pred_clf, target_names=class_names, output_dict=True)

    # Retrieval with CLIP + Vector DB (Chroma)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(clip_device)
    clip_model.eval()

    vector_db_path = PROJECT_DIR / "vector_db" / f"clip_chroma_db_seed_{seed}"
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
    collection = chroma_client.get_or_create_collection(
        name="clip_image_embeddings",
        metadata={"hnsw:space": "cosine"},
    )

    db_embed_t0 = time.time()
    train_emb = _clip_encode_images(train_paths, clip_model, clip_processor, clip_device, batch_size=32)
    collection.add(
        ids=[f"train_{i:06d}" for i in range(len(train_paths))],
        embeddings=train_emb.tolist(),
        metadatas=[
            {"label": int(train_labels[i]), "class_name": class_names[int(train_labels[i])], "path": str(train_paths[i])}
            for i in range(len(train_paths))
        ],
    )
    db_build_seconds = time.time() - db_embed_t0
    avg_retrieval_add_seconds = float(db_build_seconds / max(1, len(train_paths)))

    query_t0 = time.time()
    val_emb = _clip_encode_images(val_paths, clip_model, clip_processor, clip_device, batch_size=32)
    query_result = collection.query(query_embeddings=val_emb.tolist(), n_results=5, include=["metadatas"])
    y_pred_ret = []
    for row in query_result["metadatas"]:
        nn_labels = [int(item["label"]) for item in row]
        y_pred_ret.append(Counter(nn_labels).most_common(1)[0][0])
    y_pred_ret = np.array(y_pred_ret)
    ret_infer_seconds = time.time() - query_t0
    ret_metrics = get_metrics(val_labels, y_pred_ret)
    ret_report = classification_report(val_labels, y_pred_ret, target_names=class_names, output_dict=True)

    comparison = pd.DataFrame(
        [
            {"method": "Classifier", **clf_metrics, "total_inference_sec": clf_infer_seconds},
            {"method": "Retrieval_CLIP_VectorDB", **ret_metrics, "total_inference_sec": ret_infer_seconds},
        ]
    )
    comparison["samples"] = len(val_labels)
    comparison["avg_ms_per_sample"] = (comparison["total_inference_sec"] * 1000.0) / comparison["samples"]
    comparison["train_or_build_seconds"] = [float(train_seconds), float(db_build_seconds)]
    comparison["compute_profile"] = [
        f"trainable_params={trainable_params}, model_size_mb={model_size_mb:.2f}",
        f"embedding_dim={int(train_emb.shape[1])}, vectors={len(train_paths)}",
    ]
    comparison_path = artifacts / "sampled25_comparison_metrics.csv"
    comparison.to_csv(comparison_path, index=False)

    clf_cm = confusion_matrix(y_true, y_pred_clf)
    ret_cm = confusion_matrix(val_labels, y_pred_ret)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(clf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Classifier Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(ret_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Retrieval Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(artifacts / "comparison_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, np.array(y_prob))
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"Classifier ROC-AUC={clf_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Classifier ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts / "classifier_roc_curve.png", dpi=180)
    plt.close()

    history_df = pd.DataFrame(history.history)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_csv_path = artifacts / "sampled25_training_history.csv"
    history_df.to_csv(history_csv_path)
    best_epoch = int(history_df["val_loss"].idxmin())
    overfit_gap = history_df["accuracy"] - history_df["val_accuracy"]
    overfit_epoch = int((overfit_gap > 0.08).idxmax()) if (overfit_gap > 0.08).any() else None

    print("\nPer-epoch learning log:")
    for epoch, row in history_df.iterrows():
        gap = row["accuracy"] - row["val_accuracy"]
        print(
            f"Epoch {int(epoch):02d}: "
            f"loss={row['loss']:.4f}, acc={row['accuracy']:.4f}, auc={row['auc']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.4f}, val_auc={row['val_auc']:.4f}, "
            f"gen_gap={gap:.4f}"
        )
    print(f"Best generalization epoch by val_loss: {best_epoch}")
    if overfit_epoch is not None:
        print(f"Potential overfitting starts near epoch: {overfit_epoch}")
    else:
        print("No strong overfitting gap (>0.08) detected across epochs.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history_df.index, history_df["loss"], marker="o", label="train_loss")
    plt.plot(history_df.index, history_df["val_loss"], marker="o", label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 3, 2)
    plt.plot(history_df.index, history_df["accuracy"], marker="o", label="train_acc")
    plt.plot(history_df.index, history_df["val_accuracy"], marker="o", label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1, 3, 3)
    plt.plot(history_df.index, history_df["auc"], marker="o", label="train_auc")
    plt.plot(history_df.index, history_df["val_auc"], marker="o", label="val_auc")
    plt.legend()
    plt.title("AUC")
    plt.tight_layout()
    plt.savefig(artifacts / "training_curves.png", dpi=180)
    plt.close()

    summary = {
        "dataset": {
            "run_started_at": run_started_at,
            "seed": seed,
            "subset_root": str(data_root),
            "total_samples": int(sum(counts.values())),
            "class_counts": dict(counts),
            "height_mean": float(np.mean(h)),
            "height_std": float(np.std(h)),
            "width_mean": float(np.mean(w)),
            "width_std": float(np.std(w)),
            "classwise_stats": eda_by_class,
            "train_samples": int(len(train_labels)),
            "test_samples": int(len(val_labels)),
        },
        "classifier": {
            "train_seconds": float(train_seconds),
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "epochs_ran": int(len(history.history["loss"])),
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
            **clf_metrics,
            "roc_auc": clf_auc,
            "inference_seconds": float(clf_infer_seconds),
            "classification_report": clf_report,
        },
        "retrieval": {
            **ret_metrics,
            "inference_seconds": float(ret_infer_seconds),
            "db_build_seconds": float(db_build_seconds),
            "avg_add_image_seconds": avg_retrieval_add_seconds,
            "embedding_dim": int(train_emb.shape[1]),
            "indexed_vectors": int(len(train_paths)),
            "k": 5,
            "encoder": clip_model_name,
            "vector_db": "chroma",
            "vector_db_path": str(vector_db_path),
            "classification_report": ret_report,
        },
        "outputs": {
            "model_path": str(model_dir / "cats_vs_dogs_model_quick500.keras"),
            "comparison_csv": str(comparison_path),
            "training_history_csv": str(history_csv_path),
            "eda_class_distribution_plot": str(artifacts / "eda_class_distribution.png"),
            "eda_size_distribution_plot": str(artifacts / "eda_size_distribution.png"),
            "eda_samples_plot": str(artifacts / "eda_samples.png"),
            "training_curves_plot": str(artifacts / "training_curves.png"),
            "classifier_roc_plot": str(artifacts / "classifier_roc_curve.png"),
            "confusion_matrices_plot": str(artifacts / "comparison_confusion_matrices.png"),
        },
        "learning_dynamics": {
            "best_epoch_by_val_loss": best_epoch,
            "potential_overfit_epoch": overfit_epoch,
            "max_generalization_gap": float(overfit_gap.max()),
        },
        "compute_environment": {
            "tensorflow_gpu_available": bool(tf.config.list_physical_devices("GPU")),
            "pytorch_cuda_available": bool(torch.cuda.is_available()),
            "clip_device_used": clip_device,
        },
        "incremental_cost_estimates": {
            "deep_learning_refit_5_images_seconds": float((train_seconds / max(1, len(train_labels))) * 5),
            "deep_learning_refit_10_images_seconds": float((train_seconds / max(1, len(train_labels))) * 10),
            "retrieval_add_5_images_seconds": float(avg_retrieval_add_seconds * 5),
            "retrieval_add_10_images_seconds": float(avg_retrieval_add_seconds * 10),
        },
    }

    if query_image:
        query_path = Path(query_image)
        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")
        q_emb = _clip_encode_images([query_path], clip_model, clip_processor, clip_device, batch_size=1)
        q_res = collection.query(query_embeddings=q_emb.tolist(), n_results=5, include=["metadatas", "distances"])
        neighbors = q_res["metadatas"][0]
        distances = q_res["distances"][0]
        nn_labels = [int(item["label"]) for item in neighbors]
        pred_idx = Counter(nn_labels).most_common(1)[0][0]
        summary["retrieval_query"] = {
            "query_path": str(query_path),
            "predicted_label_idx": int(pred_idx),
            "predicted_label_name": class_names[pred_idx],
            "top_k": [
                {
                    "rank": i + 1,
                    "label_idx": int(neighbors[i]["label"]),
                    "label_name": neighbors[i]["class_name"],
                    "distance": float(distances[i]),
                    "path": neighbors[i]["path"],
                }
                for i in range(len(neighbors))
            ],
        }
    json_path = artifacts / "sampled25_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = reports / "sampled25_report.md"
    md = []
    md.append("# 25 Percent Sample Experiment Report")
    md.append("")
    md.append("## Setup")
    md.append(f"- Run started at: {run_started_at}")
    md.append(f"- Seed used: {seed}")
    md.append("- Dataset: Cats vs Dogs (clean local subset)")
    md.append("- Sampling strategy: random stratified sampling, 25% per class")
    md.append(f"- Total samples: {summary['dataset']['total_samples']}")
    md.append(f"- Train/Test split: {summary['dataset']['train_samples']} / {summary['dataset']['test_samples']}")
    md.append("- Backbone: EfficientNetB0 (ImageNet pretrained, frozen)")
    md.append("- Classification head: GlobalAveragePooling2D + Dropout(0.45) + Dense(1, sigmoid, L2)")
    md.append(f"- Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    md.append(f"- Batch size: {BATCH_SIZE}")
    md.append("- Optimizer/Loss: Adam + Binary Crossentropy")
    md.append("- Data augmentation: random flip + rotation + zoom (training only)")
    md.append("- Loss regularization: label smoothing=0.05")
    md.append("- Early stopping: monitor=val_loss, patience=3, restore_best_weights=True")
    md.append("- Learning-rate scheduling: ReduceLROnPlateau(factor=0.35, patience=1)")
    md.append("- Retrieval: CLIP image encoder + Chroma vector DB (cosine, top-5 voting)")
    md.append("")
    md.append("## Model Architecture Summary")
    md.append("```text")
    md.append(model_summary_text.rstrip())
    md.append("```")
    md.append("")
    md.append("## EDA")
    md.append(f"- Class balance: {summary['dataset']['class_counts']}")
    md.append(
        f"- Image size stats (H mean/std): {summary['dataset']['height_mean']:.1f} / {summary['dataset']['height_std']:.1f}"
    )
    md.append(
        f"- Image size stats (W mean/std): {summary['dataset']['width_mean']:.1f} / {summary['dataset']['width_std']:.1f}"
    )
    md.append("- Class-wise image statistics:")
    for cname, stat in summary["dataset"]["classwise_stats"].items():
        md.append(
            f"  - {cname}: n={stat['count']}, H(mean/std)={stat['height_mean']:.1f}/{stat['height_std']:.1f}, "
            f"W(mean/std)={stat['width_mean']:.1f}/{stat['width_std']:.1f}"
        )
    md.append("")
    md.append("## Model Performance")
    md.append(f"- Epochs trained: {summary['classifier']['epochs_ran']}")
    md.append(f"- Training time (s): {summary['classifier']['train_seconds']:.2f}")
    md.append(f"- Accuracy: {summary['classifier']['accuracy']:.4f}")
    md.append(f"- Precision: {summary['classifier']['precision']:.4f}")
    md.append(f"- Recall: {summary['classifier']['recall']:.4f}")
    md.append(f"- F1: {summary['classifier']['f1']:.4f}")
    md.append(f"- Inference time total (s): {summary['classifier']['inference_seconds']:.4f}")
    md.append(f"- ROC-AUC: {summary['classifier']['roc_auc']:.4f}")
    md.append(f"- Best learning epoch (val_loss minimum): {summary['learning_dynamics']['best_epoch_by_val_loss']}")
    md.append(f"- Potential overfitting epoch (gap > 0.08): {summary['learning_dynamics']['potential_overfit_epoch']}")
    md.append("")
    md.append("## Epoch Logs (Training History)")
    md.append("")
    history_headers = ["epoch", "loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"]
    md.append("| " + " | ".join(history_headers) + " |")
    md.append("|" + "|".join(["---"] * len(history_headers)) + "|")
    for epoch, row in history_df.iterrows():
        md.append(
            "| "
            + " | ".join(
                [
                    str(int(epoch)),
                    f"{row['loss']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['auc']:.4f}",
                    f"{row['val_loss']:.4f}",
                    f"{row['val_accuracy']:.4f}",
                    f"{row['val_auc']:.4f}",
                ]
            )
            + " |"
        )
    md.append("")
    md.append("## Training/Inference Time Breakdown")
    md.append(f"- Total training time (s): {summary['classifier']['train_seconds']:.4f}")
    md.append(f"- Classifier inference total (s): {summary['classifier']['inference_seconds']:.4f}")
    md.append(f"- Retrieval inference total (s): {summary['retrieval']['inference_seconds']:.4f}")
    md.append(f"- Retrieval DB build time (s): {summary['retrieval']['db_build_seconds']:.4f}")
    md.append(f"- Retrieval average add-one-image time (s): {summary['retrieval']['avg_add_image_seconds']:.4f}")
    md.append("")
    md.append("## Computational Cost Profile")
    md.append(f"- Classifier total parameters: {summary['classifier']['total_params']}")
    md.append(f"- Classifier trainable parameters: {summary['classifier']['trainable_params']}")
    md.append(f"- Saved model size (MB): {summary['classifier']['model_size_mb']:.2f}")
    md.append(f"- Retrieval embedding dimension: {summary['retrieval']['embedding_dim']}")
    md.append(f"- Retrieval indexed vectors: {summary['retrieval']['indexed_vectors']}")
    md.append(
        f"- TensorFlow GPU available: {summary['compute_environment']['tensorflow_gpu_available']}, "
        f"PyTorch CUDA available: {summary['compute_environment']['pytorch_cuda_available']}, "
        f"CLIP device used: {summary['compute_environment']['clip_device_used']}"
    )
    md.append("")
    md.append("## Incremental Update Cost Estimate")
    md.append(
        f"- Deep learning refit estimate (5 images): "
        f"{summary['incremental_cost_estimates']['deep_learning_refit_5_images_seconds']:.4f} s"
    )
    md.append(
        f"- Deep learning refit estimate (10 images): "
        f"{summary['incremental_cost_estimates']['deep_learning_refit_10_images_seconds']:.4f} s"
    )
    md.append(
        f"- Retrieval add estimate (5 images): "
        f"{summary['incremental_cost_estimates']['retrieval_add_5_images_seconds']:.4f} s"
    )
    md.append(
        f"- Retrieval add estimate (10 images): "
        f"{summary['incremental_cost_estimates']['retrieval_add_10_images_seconds']:.4f} s"
    )
    md.append("")
    md.append("## Retrieval Performance")
    md.append(f"- Accuracy: {summary['retrieval']['accuracy']:.4f}")
    md.append(f"- Precision: {summary['retrieval']['precision']:.4f}")
    md.append(f"- Recall: {summary['retrieval']['recall']:.4f}")
    md.append(f"- F1: {summary['retrieval']['f1']:.4f}")
    md.append(f"- Inference time total (s): {summary['retrieval']['inference_seconds']:.4f}")
    md.append(f"- Vector DB build time (s): {summary['retrieval']['db_build_seconds']:.4f}")
    md.append(f"- Encoder: {summary['retrieval']['encoder']}")
    md.append(f"- Vector DB path: {summary['retrieval']['vector_db_path']}")
    if "retrieval_query" in summary:
        md.append(f"- Query image prediction: {summary['retrieval_query']['predicted_label_name']}")
    md.append("")
    md.append("## Comparative Table")
    md.append("")
    headers = [
        "method",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "total_inference_sec",
        "samples",
        "avg_ms_per_sample",
    ]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in comparison.iterrows():
        md.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['total_inference_sec']:.4f}",
                    str(int(row["samples"])),
                    f"{row['avg_ms_per_sample']:.4f}",
                ]
            )
            + " |"
        )
    md.append("")
    md.append("## Output Files")
    md.append(f"- `{json_path}`")
    md.append(f"- `{comparison_path}`")
    md.append(f"- `{history_csv_path}`")
    md.append(f"- `{artifacts / 'eda_class_distribution.png'}`")
    md.append(f"- `{artifacts / 'eda_size_distribution.png'}`")
    md.append(f"- `{artifacts / 'eda_samples.png'}`")
    md.append(f"- `{artifacts / 'training_curves.png'}`")
    md.append(f"- `{artifacts / 'classifier_roc_curve.png'}`")
    md.append(f"- `{artifacts / 'comparison_confusion_matrices.png'}`")
    md.append(f"- `{md_path}`")
    md_path.write_text("\n".join(md), encoding="utf-8")

    log_md_path = reports / "run_sampled25_log.md"
    run_log_md = []
    run_log_md.append("# Run Log")
    run_log_md.append("")
    run_log_md.append(f"- Run started at: {run_started_at}")
    run_log_md.append(f"- Seed: {seed}")
    run_log_md.append(f"- Train samples: {summary['dataset']['train_samples']}")
    run_log_md.append(f"- Validation samples: {summary['dataset']['test_samples']}")
    run_log_md.append(f"- Classifier accuracy: {summary['classifier']['accuracy']:.4f}")
    run_log_md.append(f"- Retrieval accuracy: {summary['retrieval']['accuracy']:.4f}")
    run_log_md.append(f"- Best epoch: {summary['learning_dynamics']['best_epoch_by_val_loss']}")
    run_log_md.append("")
    run_log_md.append("## Epoch Learning Trace")
    run_log_md.append("")
    for epoch, row in history_df.iterrows():
        gap = row["accuracy"] - row["val_accuracy"]
        run_log_md.append(
            f"- Epoch {int(epoch):02d}: train_acc={row['accuracy']:.4f}, val_acc={row['val_accuracy']:.4f}, "
            f"train_loss={row['loss']:.4f}, val_loss={row['val_loss']:.4f}, gap={gap:.4f}"
        )
    if "retrieval_query" in summary:
        run_log_md.append("")
        run_log_md.append("## Retrieval Query Result")
        run_log_md.append("")
        run_log_md.append(f"- Query image: `{summary['retrieval_query']['query_path']}`")
        run_log_md.append(f"- Predicted class: `{summary['retrieval_query']['predicted_label_name']}`")
    log_md_path.write_text("\n".join(run_log_md), encoding="utf-8")

    print(f"Report generated: {md_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Run log markdown: {log_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enriched 25% cats vs dogs experiment.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, uses current timestamp.")
    parser.add_argument(
        "--query-image",
        type=str,
        default=None,
        help="Optional image path for CLIP+VectorDB retrieval prediction after indexing.",
    )
    args = parser.parse_args()
    selected_seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    run(selected_seed, query_image=args.query_image)
