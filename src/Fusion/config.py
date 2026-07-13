"""
Central configuration for the Fusion module (member 5).

This module fuses:
  - CV       (member 2/3): SkinLesionClassifier over a ResNet50/EfficientNet-B0
              backbone -> we tap the SPATIAL feature map (before pooling),
              i.e. a grid of "image patch" tokens.
  - NLP      (member 4):   BioBERT -> we tap the PER-TOKEN last_hidden_state
              (before mean-pooling), i.e. a sequence of "word" tokens.

Cross-attention lets image patches attend to symptom words and vice versa,
before the two modalities are pooled and classified jointly.
"""

import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


CV_CHECKPOINT_PATH = os.environ.get("CV_CHECKPOINT_PATH", os.path.join(PROJECT_ROOT, "best_model_ResNet50.pt"))
CV_BACKBONE = "resnet50"          
CV_IMAGE_SIZE = 224
CV_FREEZE_BACKBONE = True         
CV_UNFREEZE_LAST_N_BLOCKS = 0     

NLP_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
NLP_MAX_LENGTH = 128
NLP_FREEZE_BACKBONE = True        
NLP_UNFREEZE_LAST_N_LAYERS = 0    

CV_EMBEDDING_DIM = 512            
                                   
NLP_EMBEDDING_DIM = 768           


D_MODEL = 256          
N_HEADS = 8
FF_DIM = 512
N_LAYERS = 2           
DROPOUT = 0.1


CLASS_NAMES = [
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis-like Lesions (BKL)",
    "Eczema",
    "Melanocytic Nevi (NV)",
    "Melanoma",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Warts Molluscum and other Viral Infections",
]
NUM_CLASSES = len(CLASS_NAMES)


BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 2e-4
BACKBONE_LEARNING_RATE = 1e-5     
WEIGHT_DECAY = 1e-5


AUX_IMAGE_LOSS_WEIGHT = 0.3
AUX_TEXT_LOSS_WEIGHT = 0.3
FUSED_LOSS_WEIGHT = 1.0

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fusion_model.pt")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
