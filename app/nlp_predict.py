# app/nlp_predict.py
import torch
import numpy as np
import re

# Configuration: Triage Levels and Severity Keywords
TRIAGE_MAP = {
    "mild":     "✅ Home Care",
    "moderate": "✅ Home Care",
    "severe":   "⚠️ See a Doctor",
    "urgent":   "🚨 URGENT"
}

SEVERITY_KEYWORDS = {
    "urgent": ["bleeding", "cancer", "melanoma", "spreading fast", "changing mole", "urgent"],
    "severe": ["painful", "severe", "infected", "pus", "swollen", "spreading", "fever"],
    "moderate": ["weeks", "persistent", "recurring", "getting worse", "moderate"],
    "mild": ["mild", "slight", "minor", "small", "occasional", "dry", "itchy"]
}

# Model Loading: BioBERT and Classifier Initialization
def load_nlp_model():
    try:
        import joblib
        from tokenizers import BertWordPieceTokenizer
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel
        from transformers import BertModel

        import os

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkl_path = os.path.join(
            BASE_DIR,
            "models",
            "nlp",
            "symptom_classifier_biobert.pkl"
        )
        data = joblib.load(pkl_path)
        classifier = data["model"]

        MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
        vocab_path = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.txt")
        tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)
        tokenizer.enable_truncation(max_length=128)

        bert_model = BertModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
        bert_model.eval()

        print("✅ Real BioBERT NLP loaded!")
        return classifier, tokenizer, bert_model, True

    except Exception as e:
        import traceback
        print(f"⚠️ Fallback to keywords: {e}")
        traceback.print_exc()
        return None, None, None, False


# Prediction Logic: BioBERT Inference or Keyword Fallback
def predict_symptoms(text, classifier=None, tokenizer=None, bert_model=None, use_real=False):
    text_lower = text.lower()
    
    # Use actual ML model if available
    if use_real and classifier is not None:
        cleaned = re.sub(r'[^\w\s]', ' ', text_lower).strip()

        tokenizer.enable_padding(length=None)
        encodings = tokenizer.encode_batch([cleaned])
        max_len = max(len(e.ids) for e in encodings)
        input_ids = torch.tensor([e.ids + [0] * (max_len - len(e.ids)) for e in encodings])
        attention_mask = torch.tensor([
            e.attention_mask + [0] * (max_len - len(e.attention_mask))
            for e in encodings
        ])

        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            embedding = (
                torch.sum(outputs.last_hidden_state * mask, dim=1) /
                torch.clamp(mask.sum(dim=1), min=1e-9)
            ).squeeze(0).tolist()

        condition = classifier.predict([embedding])[0]
        proba = classifier.predict_proba([embedding])[0]
        confidence = round(float(proba.max()), 4)
        
   # Fallback to heuristic keyword-based classification
    else:
        keyword_map = {
            "Eczema": ["itchy", "itch", "dry", "rash", "red", "scaly"],
            "Melanoma": ["mole", "dark spot", "changing", "bleeding", "asymmetry"],
            "Psoriasis pictures Lichen Planus and related diseases": ["silver", "plaque", "psoriasis"],
            "Atopic Dermatitis": ["atopic", "allergic", "inflamed"],
            "Basal Cell Carcinoma (BCC)": ["pearly", "bump", "sore", "ulcer"],
            "Tinea Ringworm Candidiasis and other Fungal Infections": ["ring", "fungal", "ringworm"],
            "Melanocytic Nevi (NV)": ["mole", "nevus", "birthmark"],
            "Benign Keratosis-like Lesions (BKL)": ["rough", "waxy", "keratosis"],
            "Seborrheic Keratoses and other Benign Tumors": ["seborrheic", "greasy"],
            "Warts Molluscum and other Viral Infections": ["wart", "molluscum", "viral"]
        }
        scores = {c: sum(1 for kw in kws if kw in text_lower)
                  for c, kws in keyword_map.items()}
        condition = max(scores, key=scores.get)
        best_score = scores[condition]
        confidence = round(min(0.5 + best_score * 0.1, 0.95), 2) if best_score > 0 else 0.3

   # Severity analysis based on keywords
    severity = "mild"
    for sev, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            severity = sev
            break
   # Override severity for high-risk conditions
    if "melanoma" in condition.lower() or "bleeding" in text_lower:
        severity = "urgent"

    return {
        "condition": condition,
        "confidence": confidence,
        "severity": severity,
        "triage": TRIAGE_MAP[severity],
        "embedding": [0.0] * 768
    }
    