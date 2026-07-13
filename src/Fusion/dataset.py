import os
import csv
import random

from PIL import Image
from torch.utils.data import Dataset

import config


class PairedSkinScanDataset(Dataset):
    def __init__(self, manifest_csv, image_transform):

        self.rows = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        self.image_transform = image_transform
        self.label_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.image_transform(image)
        text = row["text"]
        label = self.label_to_idx[row["condition"]]
        return image, text, label


def collate_fn(batch):

    import torch
    images, texts, labels = zip(*batch)
    return torch.stack(images, dim=0), list(texts), torch.tensor(labels, dtype=torch.long)



_SYMPTOM_TEMPLATES = {
    "Melanoma": ["a dark mole that has been growing and bleeding", "an irregular black spot with uneven borders"],
    "Eczema": ["a dry itchy red patch on my elbow", "flaky irritated skin that won't stop itching"],
    "Atopic Dermatitis": ["chronic itchy rash behind my knees", "recurring dry scaly patches since childhood"],
    "Basal Cell Carcinoma (BCC)": ["a pearly bump that won't heal", "a shiny nodule that occasionally bleeds"],
    "Melanocytic Nevi (NV)": ["a small brown mole, same size for years", "a flat tan spot that's never changed"],
    "Benign Keratosis-like Lesions (BKL)": ["a rough waxy patch that looks stuck on", "a scaly tan growth on my back"],
    "Psoriasis pictures Lichen Planus and related diseases": ["thick silvery scaly plaques on my elbows", "purplish itchy flat bumps on my wrists"],
    "Seborrheic Keratoses and other Benign Tumors": ["a warty brown growth that's been there for years", "a crusty stuck-on looking spot"],
    "Tinea Ringworm Candidiasis and other Fungal Infections": ["a ring-shaped itchy rash with a clear center", "red scaly patches spreading in a ring"],
    "Warts Molluscum and other Viral Infections": ["small rough bumps on my hand", "tiny flesh-colored dome-shaped bumps"],
}


def generate_synthetic_manifest(out_csv, out_image_dir, n_per_class=5, seed=config.SEED):
    """Creates n_per_class random noise 'images' + templated text per class,
    purely so train.py has something runnable to smoke-test against."""
    random.seed(seed)
    os.makedirs(out_image_dir, exist_ok=True)

    rows = []
    idx = 0
    for label, templates in _SYMPTOM_TEMPLATES.items():
        for _ in range(n_per_class):
            img = Image.effect_noise((config.CV_IMAGE_SIZE, config.CV_IMAGE_SIZE), 40).convert("RGB")
            img_path = os.path.join("/content/drive/MyDrive/fusion", out_image_dir, f"synthetic_{idx:04d}.jpg")
            img.save(img_path)
            rows.append({
                "image_path": img_path,
                "text": random.choice(templates),
                "label": label,
            })
            idx += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows -> {out_csv}")
    return out_csv
