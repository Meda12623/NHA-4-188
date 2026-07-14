# app/streamlit_app.py
# import config
import streamlit as st
from PIL import Image
import sys
import os
import json
import tempfile
import importlib.util

# Project Paths and System Setup
APP_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_PATH)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
FUSION_PATH = os.path.join(SRC_PATH, "Fusion")

sys.path.insert(0, SRC_PATH)
sys.path.insert(0, APP_PATH)
sys.path.insert(0, FUSION_PATH)

# Import CV, NLP, and Fusion Modules
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from torchvision import transforms

# Load CV configuration
cv_config_spec = importlib.util.spec_from_file_location(
    "cv_config",
    os.path.join(SRC_PATH, "config.py")
)
config = importlib.util.module_from_spec(cv_config_spec)
cv_config_spec.loader.exec_module(config)

sys.modules["config"] = config

from model import SkinLesionClassifier

# Import NLP and Fusion components
from nlp_predict import load_nlp_model, predict_symptoms

fusion_config_spec = importlib.util.spec_from_file_location(
    "fusion_config",
    os.path.join(FUSION_PATH, "config.py")
)
fusion_config = importlib.util.module_from_spec(fusion_config_spec)
fusion_config_spec.loader.exec_module(fusion_config)
sys.modules["config"] = fusion_config

fusion_predict_spec = importlib.util.spec_from_file_location(
    "fusion_predict",
    os.path.join(FUSION_PATH, "predict.py")
)
fusion_predict_module = importlib.util.module_from_spec(fusion_predict_spec)
fusion_predict_spec.loader.exec_module(fusion_predict_module)
FusionPredictor = fusion_predict_module.FusionPredictor

sys.modules["config"] = config

# Streamlit Page Configuration
st.set_page_config(
    page_title="Skin Condition Detector",
    page_icon="🔬",
    layout="centered"
)

# Model Loading Functions (Cached)
@st.cache_resource
def load_cv_model():
    model = SkinLesionClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=False
    )
    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "best_model_ResNet50.pt"
    )
    state = torch.load(model_path, map_location=config.DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_nlp():
    classifier, tokenizer, bert_model, use_real = load_nlp_model()
    return classifier, tokenizer, bert_model, use_real
@st.cache_resource
def load_fusion():
    fusion_model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "Fusion",
        "best_fusion_model.pt"
    )
    return FusionPredictor(fusion_model_path)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

# Prediction Logic: Computer Vision (CV)
def predict_cv(image, model):
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        logits, embedding = model(tensor, return_embedding=True)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_class = config.CLASS_NAMES[top_idx]
    top_conf = probs[top_idx].item()

    all_scores = {
        config.CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(len(config.CLASS_NAMES))
    }

    melanoma_score = all_scores.get("melanoma", 0)
    if melanoma_score > 0.55:
        urgency = "🚨 URGENT — Please consult a dermatologist immediately!"
    elif top_conf > 0.7:
        urgency = "⚠️ See a dermatologist for confirmation."
    else:
        urgency = "✅ Monitor symptoms. See a doctor if condition worsens."

    return {
        "class": top_class,
        "confidence": round(top_conf, 4),
        "all_scores": all_scores,
        "urgency": urgency,
        "embedding": embedding[0].cpu().tolist()
    }

# ══════════════════════════════════════
# UI Components: Inputs (Image & Audio/Text)
# ══════════════════════════════════════

st.title("🔬 Skin Condition Detection System")
st.markdown("Upload a skin image and describe your symptoms to get an AI-powered assessment.")
st.divider()

cv_model = load_cv_model()
nlp_classifier, nlp_tokenizer, nlp_bert, use_real = load_nlp()
fusion_predictor = load_fusion()

#  Step 1 
st.subheader("📸 Step 1: Upload Skin Image")
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

st.divider()

# Step 2
st.subheader("📝 Step 2: Describe Your Symptoms")

audio = st.audio_input("🎤 Record your symptoms")

transcribed = ""

if audio:
    with st.spinner("🔄 Transcribing..."):
        try:
            from faster_whisper import WhisperModel
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.getvalue())
                tmp_path = tmp.name

            wmodel = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, info = wmodel.transcribe(tmp_path, language="en", beam_size=5)
            transcribed = " ".join([s.text for s in segments]).strip()
            os.unlink(tmp_path)

            if transcribed:
                st.success(f"✅ Transcribed: **{transcribed}**")
            else:
                st.warning("⚠️ Could not transcribe. Please speak clearly or type below.")
        except Exception as e:
            st.error(f"Error: {e}")

symptoms_text = st.text_area(
    "Or type your symptoms here...",
    value=transcribed,
    placeholder="e.g. itchy skin, redness, dry patches for 2 weeks",
    height=120,
    key="symptoms_input"
)

if transcribed and not symptoms_text:
    symptoms_text = transcribed
st.divider()
analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# UI Components: Display Results (CV, NLP, Fusion)
if analyze_btn:
    if not uploaded_file:
        st.warning("⚠️ Please upload an image first.")
    elif not symptoms_text:
        st.warning("⚠️ Please describe your symptoms.")
    else:
        with st.spinner("Analyzing..."):
            cv_result = predict_cv(image, cv_model)

            nlp_result = predict_symptoms(
                 symptoms_text,
                 nlp_classifier,
                 nlp_tokenizer,
                 nlp_bert,
                  use_real
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            fusion_result = fusion_predictor.predict(tmp_path, symptoms_text)
            os.unlink(tmp_path)

        st.divider()

        # ─── CV Results ───
        st.subheader("📊 Image Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Condition", cv_result["class"].replace("_", " ").title())
        with col2:
            st.metric("Confidence", f"{cv_result['confidence'] * 100:.1f}%")

        st.markdown("#### 🔢 All Condition Scores")
        for cls, score in sorted(cv_result["all_scores"].items(),
                                  key=lambda x: x[1], reverse=True):
            label = cls.replace("_", " ").title()
            st.progress(score, text=f"{label}: {round(score * 100, 1)}%")

        st.divider()

        # ─── NLP Results ───
        st.subheader("🧠 Symptom Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Condition from Symptoms", nlp_result["condition"])
        with col2:
            st.metric("NLP Confidence", f"{nlp_result['confidence'] * 100:.1f}%")

        if nlp_result["confidence"] < 0.4:
            st.warning("⚠️ Symptom description is too general. Please add more specific details.")
        else:
            st.info(f"📋 Severity: {nlp_result['severity'].title()} - {nlp_result['triage']}")

        st.divider()

        # ─── Fusion Results ───
        st.subheader("🔀 Fusion Analysis (Image + Symptoms)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Condition", fusion_result["condition"])
        with col2:
            st.metric("Fusion Confidence", f"{fusion_result['confidence'] * 100:.1f}%")

        gate = fusion_result["image_vs_text_gate"]
        if gate > 0.6:
            st.info(f"🖼️ Model leaned more on the image (gate={gate:.2f})")
        elif gate < 0.4:
            st.info(f"📝 Model leaned more on the symptoms (gate={gate:.2f})")
        else:
            st.info(f"⚖️ Model used both equally (gate={gate:.2f})")

        st.divider()

       
        st.subheader("🏥 Final Recommendation")
        urgency = cv_result["urgency"]
        if "URGENT" in urgency:
            st.error(urgency)
        elif "See" in urgency:
            st.warning(urgency)
        else:
            st.success(urgency)

        st.divider()
        st.caption("⚠️ This is an AI screening tool only. Always consult a dermatologist for diagnosis.")


# Evaluation Metrics

st.divider()
st.subheader("📈 Model Evaluation Metrics")

metrics_path = os.path.join(APP_PATH, '..', 'evaluation', 'outputs', 'metrics.json')

if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy",  f"{metrics['accuracy']  * 100:.1f}%")
    with col2:
        st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    with col3:
        st.metric("Recall",    f"{metrics['recall']    * 100:.1f}%")
    with col4:
        st.metric("F1 Score",  f"{metrics['f1_score']  * 100:.1f}%")

    cm_path = os.path.join(APP_PATH, '..', 'evaluation', 'outputs', 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
else:
    st.info("ℹ️ Run evaluation.py first to see metrics here.")