
import streamlit as st
import joblib
from preprocess import spacy_preprocess

# Load model
model = joblib.load("toxic_model_1.pkl")
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.title("Toxic Comment Classifier")

text_input = st.text_area("Enter a comment:")

if st.button("Classify"):
    if text_input:
        processed = spacy_preprocess(text_input)
        result = model.predict([processed])
        for i, label in enumerate(labels):
            st.write(f"{label}: {'✅' if result[0][i] else '❌'}")
    else:
        st.warning("Please enter some text.")
=======
# app.py
import streamlit as st
import pickle
import numpy as np
from preprocess import spacy_preprocess

# Load model and labels
with open("toxic_model_1.pkl", "rb") as f:
    model = pickle.load(f)

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Streamlit UI
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("💬 Toxic Comment Classifier")
st.markdown("Enter a comment below to check for toxic labels.")

user_input = st.text_area("📝 Enter your comment here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = spacy_preprocess(user_input)
        prediction = model.predict([cleaned])[0]

        st.subheader("🔍 Prediction Results:")
        for label, is_present in zip(LABELS, prediction):
            status = "✅ Safe" if is_present == 0 else "⚠️ Toxic"
            st.write(f"**{label.capitalize()}**: {status}")
