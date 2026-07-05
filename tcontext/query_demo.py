import argparse
from collections import Counter
from pathlib import Path

import chromadb
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DB_ROOT = PROJECT_DIR / "vector_db"
COLLECTION_NAME = "clip_image_embeddings"
CLIP_NAME = "openai/clip-vit-base-patch32"


def _load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    model.eval()
    return model, processor, device


def _encode_images(paths, model, processor, device):
    images = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        if hasattr(emb, "image_embeds"):
            emb = emb.image_embeds
        elif hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def _open_collection():
    VECTOR_DB_ROOT.mkdir(parents=True, exist_ok=True)
    db_candidates = sorted(
        [p for p in VECTOR_DB_ROOT.glob("clip_chroma_db*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    db_path = db_candidates[0] if db_candidates else VECTOR_DB_ROOT / "clip_chroma_db"
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_image(image_path: Path, label: str):
    if label not in {"cats", "dogs"}:
        raise ValueError("Label must be one of: cats, dogs")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, processor, device = _load_clip()
    collection = _open_collection()
    emb = _encode_images([image_path], model, processor, device)

    label_idx = 0 if label == "cats" else 1
    doc_id = f"manual_{label}_{image_path.stem}_{int(torch.randint(0, 10_000_000, (1,)).item())}"
    collection.add(
        ids=[doc_id],
        embeddings=emb.tolist(),
        metadatas=[{"label": label_idx, "class_name": label, "path": str(image_path)}],
    )
    print(f"Added to vector DB: id={doc_id}, label={label}, path={image_path}")


def query_image(image_path: Path, top_k: int = 5):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, processor, device = _load_clip()
    collection = _open_collection()
    if collection.count() == 0:
        raise RuntimeError("Vector database is empty. Run experiment first or add images with --add-image.")

    emb = _encode_images([image_path], model, processor, device)
    result = collection.query(query_embeddings=emb.tolist(), n_results=top_k, include=["metadatas", "distances"])
    neighbors = result["metadatas"][0]
    distances = result["distances"][0]
    nn_labels = [int(item["label"]) for item in neighbors]
    pred_idx = Counter(nn_labels).most_common(1)[0][0]
    pred_name = "cats" if pred_idx == 0 else "dogs"

    print(f"Query: {image_path}")
    print(f"Predicted label: {pred_name}")
    print("")
    print("Top neighbors:")
    for i, (meta, dist) in enumerate(zip(neighbors, distances), start=1):
        print(
            f"{i}. class={meta['class_name']}, distance={dist:.4f}, path={meta['path']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Demo tool for CLIP + Chroma incremental retrieval.")
    parser.add_argument("--query-image", type=str, default=None, help="Run retrieval prediction for this image.")
    parser.add_argument("--add-image", type=str, default=None, help="Add a new image into vector DB memory.")
    parser.add_argument("--label", type=str, default=None, help="Label for --add-image. Use 'cats' or 'dogs'.")
    parser.add_argument("--top-k", type=int, default=5, help="Neighbors for retrieval query.")
    args = parser.parse_args()

    if args.add_image:
        if not args.label:
            raise ValueError("--label is required with --add-image")
        add_image(Path(args.add_image), args.label)
    elif args.query_image:
        query_image(Path(args.query_image), top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
