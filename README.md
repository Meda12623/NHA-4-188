# 🔬 SkinScan AI — Multimodal Skin Disease Detection System

An AI system that combines **computer vision** (skin image analysis), **NLP** (patient symptom analysis), and a **fusion engine** to give a more complete, explainable skin-condition assessment than either modality alone.

> ⚠️ **Disclaimer:** This project is a screening/educational tool only. It is **not** a medical diagnostic device. Always consult a licensed dermatologist for an actual diagnosis.

---

## 🧭 Overview

A user uploads a photo of a skin lesion and describes their symptoms (by typing or recording audio). The system then:

1. Classifies the **image** into one of 10 skin conditions (CV pipeline).
2. Classifies the **symptom description** using a BioBERT-based NLP model.
3. **Fuses** both signals to produce a final condition + confidence + triage recommendation.
4. Displays live evaluation metrics (accuracy, precision, recall, F1, confusion matrix).

## ✨ Features

- 📸 Image upload + classification (EfficientNet-B0 / ResNet50 backbones)
- 🎤 Voice-to-text symptom input (Whisper) or free-text entry
- 🧠 Symptom classification via a fine-tuned BioBERT model
- 🔀 Fusion engine that weighs image vs. text evidence (gate score)
- 🩺 Urgency/triage messaging (e.g. flags likely melanoma cases)
- 📊 Live model evaluation dashboard inside the Streamlit app
- 🔍 Explainability support (Grad-CAM, prediction/error/confidence logs)
- 📈 Experiment tracking via MLflow, hyperparameter tuning via Optuna

## 🏗️ Architecture

```
                ┌─────────────────┐        ┌──────────────────┐
   Image ─────▶ │   CV Pipeline    │        │   NLP Pipeline    │ ◀───── Symptoms
                │ (EfficientNet/   │        │   (BioBERT)       │        (text/voice)
                │  ResNet50)       │        │                   │
                └────────┬─────────┘        └─────────┬─────────┘
                         │  embedding                  │  embedding
                         └───────────┬─────────────────┘
                                     ▼
                            ┌──────────────────┐
                            │  Fusion Engine    │
                            │ (gated combiner)  │
                            └────────┬──────────┘
                                     ▼
                         Final condition + confidence
                          + urgency / triage message
```

The CV model exposes both class logits **and** a fixed-size embedding vector, so the Fusion engine can consume it regardless of which backbone (EfficientNet-B0 or ResNet50) is active.

## 🗂️ Project Structure

```
NHA-4-188/
│
├── app/
│   ├── streamlit_app.py       # Main Streamlit UI (image + symptoms → fused result)
│   └── nlp_predict.py         # NLP inference helper
│
├── src/
│   ├── config.py              # Central config (paths, hyperparams, class names)
│   ├── dataset.py             # PyTorch ImageFolder-based data loaders
│   ├── transforms.py          # Resize/normalize transforms
│   ├── augmentation.py        # Class-balancing augmentation (offline)
│   ├── model.py                # SkinLesionClassifier (EfficientNet-B0 / ResNet50)
│   ├── trainer.py / trainer_mlflow.py  # Training loops (plain / MLflow-tracked)
│   ├── predict.py             # Single-image inference
│   ├── split_dataset.py       # Train/val/test splitting
│   ├── nlp/                   # Symptom dataset generation
│   └── Explainability/        # Grad-CAM notebook
│
├── data/
│   ├── symptom_dataset_v2.csv
│   └── Explainability/        # predictions.csv, errors.csv, confidence.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── nlp_training.ipynb
│   └── Explainability.ipynb
│
├── models/
│   ├── best_model.pt
│   ├── best_model_ResNet50.pt
│   └── nlp/                   # BioBERT symptom classifier + metadata
│
├── evaluation/
│   ├── evaluation.py
│   ├── optuna_tuning.py
│   ├── mlflow.db / mlruns/
│   └── outputs/                # metrics.json, confusion_matrix.png, best_config.json
│
├── outputs/figures/
├── System Design.pdf           # Full architecture & UML diagrams
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/nhahub/NHA-4-188.git
cd NHA-4-188
pip install -r requirements.txt
```

Download the processed image dataset and place `train/`, `val/`, `test/` folders inside `data/processed/` (PyTorch `ImageFolder` format — one subfolder per class).

> **Note:** `app/streamlit_app.py` currently references some hardcoded local paths (`SRC_PATH`, `FUSION_PATH`, model paths) — update these to match your local machine or repo layout before running the app.

## ▶️ Usage

**Run the Streamlit app:**

```bash
streamlit run app/streamlit_app.py
```

**Train the CV model:**

```python
import sys
sys.path.append("src")
from trainer import train_model   # or trainer_mlflow for tracked runs

train_model()
```

**Run evaluation:**

```bash
python evaluation/evaluation.py
```

**Run hyperparameter tuning:**

```bash
python evaluation/optuna_tuning.py
```

## 🧬 Skin Condition Classes (10)

`atopic_dermatitis` · `basal_cell_carcinoma_bcc` · `benign_keratosis_like_lesions_bkl` · `eczema` · `melanocytic_nevi_nv` · `melanoma` · `psoriasis_lichen_planus_and_related` · `seborrheic_keratoses_and_other_benign_tumors` · `tinea_ringworm_candidiasis_and_other_fungal_infections` · `warts_molluscum_and_other_viral_infections`

## 📊 Current Evaluation

Metrics are tracked in `evaluation/outputs/metrics.json` and rendered live in the Streamlit dashboard (accuracy, precision, recall, F1, confusion matrix). Re-run `evaluation/evaluation.py` after training to refresh these numbers.

## 👥 Team & Components

This is a 6-member collaborative system, split by component:

| Component | Responsibility |
|---|---|
| Data | Dataset curation, cleaning, splitting |
| CV | Image classification (EfficientNet-B0 / ResNet50) |
| NLP | Symptom classification (BioBERT) |
| Explainability | Grad-CAM, prediction/error analysis |
| Fusion | Combining CV + NLP signals |
| Optimization / Streamlit | Hyperparameter tuning, app UI |

See `System Design.pdf` for full UML diagrams (ERD, DFD, class, sequence, activity, state, component, deployment).

## 🛠️ Tech Stack

PyTorch · torchvision · Transformers (BioBERT) · Streamlit · Albumentations · OpenCV · scikit-learn · MLflow · Optuna · faster-whisper

## Contact ☎️
For any questions or inquiries, please feel free to reach out through the following channels:
- Mina Ibrahim  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:minaibrahim365@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mina-ibrahim-ab7472313/)
- Dina Raslan  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dinvrvslvn@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dinaraslan/)
- Shrouk Ashraf  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shroukkabeel12345@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shroukkabeel12/)
- Hend Ghallab  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hendghalabdarderali@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hend-ghallab-darder-a5520a317/)
- Shahd Elsawy  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:4ahdelsawy3@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shahdelsawy/)
- Jessica John  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:2300616@student.eelu.edu.eg) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jessica-john-cs/)
