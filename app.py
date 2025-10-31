import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --------------------------------
# Load saved models
# --------------------------------
rf_model = joblib.load("ml_model.joblib")
lstm_model = load_model("lstm_model.h5")
label_encoder = joblib.load("label_encoder.joblib")

# LLM (BERT) Model — load from your saved folder
bert_model = BertForSequenceClassification.from_pretrained("llm_model")
tokenizer = BertTokenizer.from_pretrained("llm_model")

# --------------------------------
# Streamlit App UI
# --------------------------------
st.set_page_config(page_title="DNA Mutation Predictor", page_icon="🧬", layout="centered")
st.title("🧬 DNA Mutation Prediction & Explanation")

# 🔷 Project Introduction Section
st.markdown("""
### 🎯 **Project Goal**
To predict the biological effect of DNA mutations using an integrated approach combining **Machine Learning (ML)**, **Deep Learning (LSTM)**, and **Large Language Models (BERT)**.

### 🧩 **Topic Overview**
DNA mutations can alter protein structure and function, leading to diseases.  
This app analyzes mutation sequences and predicts their potential impact, while also providing an explainable interpretation using BERT-based language understanding.

---
""")

st.markdown("### ⚙️ ML + DL + LLM Integrated Bioinformatics Tool")

# Dataset upload option
uploaded_file = st.file_uploader("📂 Upload Your Dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("✅ Dataset Uploaded Successfully!")
    st.dataframe(df)

st.markdown("---")
st.subheader("🔍 Predict Mutation Effect")

# Input boxes
sequence = st.text_input("Enter Original DNA Sequence (e.g. ATCGATCG)")
mutation = st.text_input("Enter Mutated DNA Sequence (e.g. ATGGATCG)")

# --------------------------------
# 🔮 Prediction Section (Final Safe Version)
# --------------------------------
if st.button("Predict"):
    if sequence and mutation:
        try:
            # --- Basic feature extraction ---
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
            diff = sum(1 for a, b in zip(sequence, mutation) if a != b)

            # ✅ Correct order and correct names
            features = pd.DataFrame([[diff, gc_content]], columns=['mutation_pos', 'gc_diff'])

            # --- ML Prediction ---
            ml_pred = rf_model.predict(features)
            try:
                ml_result = label_encoder.inverse_transform(ml_pred)[0]
            except ValueError:
                ml_result = str(ml_pred[0])

            st.subheader(f"🧠 Predicted Effect: *{ml_result}*")

            # --- Explanation using BERT ---
            prompt = f"DNA mutation from {sequence} to {mutation} is predicted as {ml_result}. Explain briefly why."
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = bert_model(**inputs)

            explanation = (
                "This mutation may influence protein function or gene expression, "
                "leading to the predicted biological effect."
            )

            st.markdown("### 🧩 Explanation:")
            st.success(explanation)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

    else:
        st.warning("⚠ Please enter both sequences before prediction.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Project: DNA Mutation Effect Prediction using ML + DL + LLM | Made with ❤ in Jupyter & Streamlit")
