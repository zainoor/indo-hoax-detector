import streamlit as st
import joblib
import re
from langdetect import detect
from summa.summarizer import summarize
import nltk

# --- Streamlit Page Config ---
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")

# Ensure tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# --- Helpers ---
def clean_text(text):
    return text.lower()

def is_valid_input(text):
    text = text.strip()
    return len(text) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

def fix_summary_capitalization(text):
    return ". ".join(sentence.strip().capitalize() for sentence in text.split(".") if sentence).strip() + "."

# --- Load Model & Vectorizer ---
@st.cache_resource
def load_model():
    return joblib.load("models/hoax_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("models/vectorizer.pkl")

model = load_model()
vectorizer = load_vectorizer()

# --- UI Header ---
st.markdown("<h1 style='text-align: center;'>Deteksi Berita Hoaks 🔎</h1>", unsafe_allow_html=True)


# --- Input Section ---
# --- Enhanced Input Section with Styling ---
st.markdown("""
    <style>
    .custom-box {
        background-color: #ffffff;
        border: 2px solid #b52f2f;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #f9f9f9 !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 16px !important;
    }
    .stButton > button {
        background-color: #b52f2f;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        margin-top: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ebebeb;
        border: 2px solid #b52f2f;
        transform: scale(1.04);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='custom-box'>
    <h3>📰 Masukkan Artikel Berita</h3>
""", unsafe_allow_html=True)

text = st.text_area(
    label="",
    height=200,
    placeholder="Petunjuk:\nMasukkan teks dari sumber online\nGunakan Bahasa Indonesia\nMinimal 30 karakter",
    label_visibility="collapsed"
)

submit = st.button("🔍 Periksa")

st.markdown("</div>", unsafe_allow_html=True)


# --- Process Prediction ---
if submit:
    text = text.strip()
    if not text:
        st.warning("⚠️ Tolong masukkan teks terlebih dahulu.")
    elif not is_valid_input(text):
        st.warning("⚠️ Teks terlalu pendek atau tidak valid. Masukkan minimal 30 karakter yang bermakna.")
    else:
        try:
            if detect(text) != "id":
                st.warning("⚠️ Artikel harus menggunakan Bahasa Indonesia.")
            else:
                cleaned = clean_text(text)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                confidence = proba[prediction]

                st.markdown("## Hasil Deteksi")
                result_label = "🚨 **Hoax**" if prediction == 1 else "✅ **Valid**"
                st.success(f"Hasil Deteksi: {result_label}")
                st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%}")

                if confidence < 0.60:
                    st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")

                # Most influential words
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                nonzero_idx = vectorized.nonzero()[1]
                word_scores = {feature_names[i]: coefs[i] for i in nonzero_idx}
                sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                top_words = [word for word, _ in sorted_words[:5]]

                st.markdown("### **Kata Paling Berpengaruh 🗝️**")
                st.markdown(f"<span style='font-size: 20px'>{', '.join(top_words)}</span>", unsafe_allow_html=True)

                # Article summary
                st.markdown("### Ringkasan Artikel:")
                try:
                    raw_summary = summarize(cleaned, words=80)
                    summary_text = fix_summary_capitalization(raw_summary)
                    if summary_text.strip():
                        st.info(summary_text)
                    else:
                        st.info("Teks terlalu pendek untuk diringkas secara otomatis.")
                except Exception as e:
                    st.warning(f"Gagal membuat ringkasan: {e}")

        except Exception as e:
            st.warning(f"Terjadi kesalahan saat memproses teks: {e}")

# --- Footer & Info ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.markdown("""---""")

with st.expander("Tentang Aplikasi ℹ️", expanded=False):
    st.markdown("""
    Aplikasi ini digunakan untuk mendeteksi apakah sebuah artikel mengandung informasi hoaks atau tidak berdasarkan teks yang dimasukkan.
    """)

st.markdown(
    """
    <hr style='border-top: 1px solid #bbb;'>
    <div style='text-align: center; font-size: 14px;'>
        Dibuat oleh <b>Mohammad Ramadhan Zainoor</b> · © 2025 · <a href="https://github.com/zainoor" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)
