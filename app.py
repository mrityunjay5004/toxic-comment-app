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