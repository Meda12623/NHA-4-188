"""
Central configuration for the CV pipeline.
Every other module (dataset, transforms, model, trainer, evaluate, predict)
should import its settings from here instead of hardcoding values.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Image / Data
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True          # set True when training on CUDA
PERSISTENT_WORKERS = True  # keeps workers alive between epochs (faster)
DROP_LAST_TRAIN = True     # avoids tiny/uneven last batch during training

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "ResNet50"   # switch to "resnet50" to try ResNet50 instead
NUM_CLASSES = 10
EMBEDDING_DIM = 512               # what Fusion (member 5) will consume
PRETRAINED = True
FREEZE_BACKBONE = False            # True = only train the classifier head first

# Model checkpoint path includes the backbone name, so switching MODEL_NAME
# (e.g. efficientnet_b0 -> resnet50) doesn't overwrite a previous run's best model.
BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"best_model_{MODEL_NAME}.pt")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Class names (fill in with your real skin-disease classes, in ImageFolder order)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "atopic_dermatitis",
    "basal_cell_carcinoma_bcc",
    "benign_keratosis_like_lesions_bkl",
    "eczema",
    "melanocytic_nevi_nv",
    "melanoma",
    "psoriasis_pictures_lichen_planus_and_related_diseases",
    "seborrheic_keratoses_and_other_benign_tumors",
    "tinea_ringworm_candidiasis_and_other_fungal_infections",
    "warts_molluscum_and_other_viral_infections",
]
