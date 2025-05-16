import streamlit as st
import joblib
import PyPDF2
import os
import re
from langdetect import detect

# Basic cleaning (replace with your actual cleaner if available)
def clean_text(text):
    return text.lower()

# Validation: ignore meaningless short input like "aaa"
def is_valid_input(text):
    text = text.strip()
    if len(text) < 20:  # Length check
        return False
    if not re.search(r'[a-zA-Z]{3,}', text):  # At least some real words
        return False
    return True

# Load model & vectorizer
model = joblib.load(os.path.join("models", "hoax_model.pkl"))
vectorizer = joblib.load(os.path.join("models", "vectorizer.pkl"))

# Streamlit app layout
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.title("Deteksi Berita Hoaks")

input_type = st.radio("Choose input type:", ["Text", "PDF"])
text = ""

if input_type == "Text":
    text = st.text_area("Enter article text below:")
elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

if st.button("Check"):
    if not text.strip():
        st.warning("Tolong masukkan Text atau PDF anda.")
    elif len(text.strip()) < 30:
        st.warning("Text anda terlalu pendek. Masukan minimum 30 Karakter.")
    else:
        try:
            language = detect(text)
            if language != "id":
                st.warning("Pastikan Arikel Anda menggunakan bahasa Indonesia")
            else:
                cleaned_text = clean_text(text)
                vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized)[0]
                result = "🚨 Hoax" if prediction == 1 else "✅ Valid"
                st.success(f"Result: {result}")

        except:
            st.warning("Could not detect language. Please input a valid sentence.")
