import os
import base64
import torch
import streamlit as st
import numpy as np
from PIL import Image

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    ViTForImageClassification, ViTImageProcessor
)

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Multimodal Detection", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_BERT = os.path.join(ROOT_DIR, "models", "bert_detector")
MODEL_ROBERTA = os.path.join(ROOT_DIR, "models", "roberta_detector")
MODEL_VIT = os.path.join(ROOT_DIR, "vit_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# HEADER (LOGO QNU)
# ===============================
logo_path = os.path.join(BASE_DIR, "logo_qnu.jpg")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px;">
        <img src="data:image/png;base64,{logo_base64}" width="70">
        <div>
            <div style="font-size:18px; font-weight:bold;">
                Quy Nhon University
            </div>
            <div style="font-size:14px;">
                Information Technology
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    bert_tok = DistilBertTokenizer.from_pretrained(MODEL_BERT)
    bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_BERT).to(DEVICE).eval()

    rob_tok = RobertaTokenizer.from_pretrained(MODEL_ROBERTA)
    rob_model = RobertaForSequenceClassification.from_pretrained(MODEL_ROBERTA).to(DEVICE).eval()

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    vit_model.load_state_dict(torch.load(MODEL_VIT, map_location=DEVICE))
    vit_model.to(DEVICE).eval()

    return bert_tok, bert_model, rob_tok, rob_model, processor, vit_model

bert_tok, bert_model, rob_tok, rob_model, processor, vit_model = load_models()

# ===============================
# LABEL
# ===============================
label_text = {0: "Human", 1: "AI"}
label_img = {0: "REAL", 1: "FAKE"}

# ===============================
# PREDICT
# ===============================
def predict_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs

def predict_image(image):
    enc = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = vit_model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    # 🔥 FIX ĐẢO LABEL
    pred = 1 - pred

    return pred, probs

# ===============================
# TABLE UI
# ===============================
def render_table(results):
    st.markdown("### Results")

    for name, label, conf in results:
        col1, col2, col3 = st.columns([2,2,4])

        with col1:
            st.write(name)

        with col2:
            st.write(label)

        with col3:
            st.progress(float(conf)/100)

        st.caption(f"{conf:.2f}%")

# ===============================
# FINAL DECISION
# ===============================
def final_decision(results):
    return max(results, key=lambda x: x[2])[1]

# ===============================
# MAIN UI
# ===============================
st.title("Multimodal AI Detection System")

mode = st.radio("Mode", ["Text", "Image", "Fusion"])

# ===============================
# TEXT
# ===============================
if mode == "Text":
    text = st.text_area("Input text")

    if st.button("Run"):
        if text.strip():
            p1, prob1 = predict_text(bert_model, bert_tok, text)
            p2, prob2 = predict_text(rob_model, rob_tok, text)

            results = [
                ("BERT", label_text[p1], prob1[p1]*100),
                ("RoBERTa", label_text[p2], prob2[p2]*100)
            ]

            render_table(results)
            st.markdown(f"**Final:** {final_decision(results)}")
        else:
            st.warning("Please enter text")

# ===============================
# IMAGE
# ===============================
elif mode == "Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Run"):
            p, prob = predict_image(img)

            results = [
                ("ViT", label_img[p], prob[p]*100)
            ]

            render_table(results)

# ===============================
# FUSION
# ===============================
elif mode == "Fusion":
    text = st.text_area("Text")
    file = st.file_uploader("Image", type=["jpg","png","jpeg"])

    if text and file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Run Fusion"):
            p1, prob1 = predict_text(bert_model, bert_tok, text)
            p2, prob2 = predict_text(rob_model, rob_tok, text)
            p3, prob3 = predict_image(img)

            results = [
                ("BERT", label_text[p1], prob1[p1]*100),
                ("RoBERTa", label_text[p2], prob2[p2]*100),
                ("ViT", label_img[p3], prob3[p3]*100)
            ]

            render_table(results)
            st.markdown(f"**Final Decision:** {final_decision(results)}")

st.markdown("---")
st.caption("Multimodal AI Detection")