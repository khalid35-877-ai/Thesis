from pathlib import Path
import shutil
import zipfile

from PIL import Image
import tensorflow as tf


def _find_local_tfds_zip() -> Path:
    downloads_dir = Path.home() / "tensorflow_datasets" / "downloads" / "cats_vs_dogs"
    zips = sorted(downloads_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(
            "No local cats_vs_dogs zip found in tensorflow_datasets cache. "
            "Please run a tfds cats_vs_dogs download once."
        )
    return zips[0]


def _extract_raw_dataset(raw_root: Path) -> Path:
    petimages = raw_root / "PetImages"
    if petimages.exists():
        return petimages

    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = _find_local_tfds_zip()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_root)
    return petimages


def _build_clean_subset(raw_petimages: Path, clean_root: Path, max_per_class: int = 2500) -> Path:
    cats_dir = clean_root / "cats"
    dogs_dir = clean_root / "dogs"
    marker = clean_root / f".ready_rgb_v2_{max_per_class}"
    if marker.exists() and cats_dir.exists() and dogs_dir.exists():
        return clean_root

    if clean_root.exists():
        shutil.rmtree(clean_root)
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    def collect(src_dir: Path, dst_dir: Path) -> int:
        saved = 0
        for file_path in sorted(src_dir.glob("*.jpg")):
            if saved >= max_per_class:
                break
            try:
                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    image = image.convert("RGB")
                    output_name = f"{saved:05d}.jpg"
                    image.save(dst_dir / output_name, format="JPEG", quality=95)
                saved += 1
            except Exception:
                continue
        return saved

    cat_count = collect(raw_petimages / "Cat", cats_dir)
    dog_count = collect(raw_petimages / "Dog", dogs_dir)
    if cat_count == 0 or dog_count == 0:
        raise RuntimeError("Failed to prepare cleaned cats/dogs images from local archive.")

    marker.write_text(f"cats={cat_count},dogs={dog_count}\n", encoding="utf-8")
    return clean_root


def get_dataset_root() -> Path:
    data_root = Path(__file__).resolve().parent / "data"
    raw_petimages = _extract_raw_dataset(data_root / "raw")
    clean_root = _build_clean_subset(raw_petimages, data_root / "clean", max_per_class=2500)
    return clean_root


def make_image_ds(subset: str, validation_split: float, batch_size: int, image_size=(224, 224), seed: int = 123):
    root = get_dataset_root()
    ds = tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
    )
    return ds
