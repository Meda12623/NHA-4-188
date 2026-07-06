"""
Splits a raw dataset (one folder per class, images directly inside) into
train/val/test folders while preserving the class subfolder structure,
so the result is directly compatible with torchvision.datasets.ImageFolder
(and therefore with dataset.py / config.py as-is).

Before:
    raw_data/
        eczema/
            img1.jpg
            img2.jpg
        melanoma/
            img1.jpg
        ...

After:
    data/processed/
        train/
            eczema/...
            melanoma/...
        val/
            eczema/...
            melanoma/...
        test/
            eczema/...
            melanoma/...
"""

import os
import re
import shutil
import random

# ---------------------------------------------------------------------------
# EDIT THESE TWO PATHS to match your machine
# ---------------------------------------------------------------------------
RAW_DATA_DIR = "D:\\datasets\\IMG_CLASSES"  # your current per-class folders
OUTPUT_DIR = os.path.join("data", "processed")  # where train/val/test will be created

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def clean_class_name(raw_name):
    """
    Turns messy folder names like:
        "1. Eczema 1677"
        "2. Melanoma 15.75k"
        "4. Basal Cell Carcinoma (BCC) 3323"
    into clean, ML-friendly class names:
        "eczema"
        "melanoma"
        "basal_cell_carcinoma_bcc"
    """
    name = raw_name

    # remove leading index like "1. " or "10. "
    name = re.sub(r'^\d+\.\s*', '', name)

    # remove trailing image-count like " 1677", " - 1.25k", " 15.75k"
    name = re.sub(r'[\s\-]*[\d,.]+k?\s*$', '', name, flags=re.IGNORECASE)

    # normalize: lowercase, non-alphanumeric -> underscore, collapse repeats
    name = re.sub(r'[^\w]+', '_', name).strip('_').lower()
    name = re.sub(r'_+', '_', name)

    return name


def split_dataset(raw_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR):
    random.seed(SEED)

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0"

    raw_class_names = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    if not raw_class_names:
        raise RuntimeError(f"No class folders found inside '{raw_dir}'")

    # map: raw folder name (on disk) -> cleaned class name (used in output)
    class_map = {raw: clean_class_name(raw) for raw in raw_class_names}

    print(f"Found {len(raw_class_names)} classes:")
    for raw, clean in class_map.items():
        print(f"  '{raw}'  ->  '{clean}'")
    print()

    clean_names = list(class_map.values())
    if len(set(clean_names)) != len(clean_names):
        raise RuntimeError(
            "Two raw folder names cleaned to the same class name — "
            "check clean_class_name() output above for collisions."
        )

    for split in ("train", "val", "test"):
        for clean in clean_names:
            os.makedirs(os.path.join(output_dir, split, clean), exist_ok=True)

    summary = {}

    for raw_cls, clean_cls in class_map.items():
        cls_dir = os.path.join(raw_dir, raw_cls)
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # whatever remains goes to test, so rounding doesn't drop images
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, split_imgs in (
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ):
            dst_dir = os.path.join(output_dir, split_name, clean_cls)
            for img_name in split_imgs:
                src_path = os.path.join(cls_dir, img_name)
                dst_path = os.path.join(dst_dir, img_name)
                shutil.copy2(src_path, dst_path)

        summary[clean_cls] = {
            "total": n_total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        print(f"{clean_cls}: total={n_total} -> train={len(train_imgs)}, "
              f"val={len(val_imgs)}, test={len(test_imgs)}")

    print(f"\nDone. Split dataset written to: {os.path.abspath(output_dir)}")
    return summary


if __name__ == "__main__":
    split_dataset()
