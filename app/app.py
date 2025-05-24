import streamlit as st
import joblib
import re
from langdetect import detect
from summa.summarizer import summarize
import nltk

nltk.download('punkt')

# --- Helpers ---
def clean_text(text):
    return text.lower()

def is_valid_input(text):
    text = text.strip()
    return len(text) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

# Capitalize summary sentences
def fix_summary_capitalization(text):
    return ". ".join(sentence.strip().capitalize() for sentence in text.split(".") if sentence).strip() + "."

# Load model & vectorizer
model = joblib.load("models/hoax_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# --- App layout ---
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.markdown("<h1 style='text-align: center;'>Deteksi Berita Hoaks</h1>", unsafe_allow_html=True)

# Input
st.markdown("##  Masukkan Artikel Berita 📝")
text = st.text_area(
    "",
    height=200,
    placeholder="Petunjuk:\n- Masukkan teks dari sumber online\n- Gunakan Bahasa Indonesia\n- Minimal 30 karakter"
)

# Predict
if st.button("🔍 Periksa"):
    if not text.strip():
        st.warning("⚠️ Tolong masukkan teks terlebih dahulu.")
    elif not is_valid_input(text):
        st.warning("⚠️ Teks terlalu pendek atau tidak valid. Masukkan minimal 30 karakter yang bermakna.")
    else:
        try:
            language = detect(text)
            if language != "id":
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

                # Add confidence threshold warning
                if confidence < 0.60:
                    st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")


                # Top words
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                nonzero_idx = vectorized.nonzero()[1]
                word_scores = {feature_names[i]: coefs[i] for i in nonzero_idx}
                sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                top_words = [word for word, _ in sorted_words[:5]]

                st.markdown("### **Kata Paling Berpengaruh 🗝️**")
                st.markdown(f"<span style='font-size: 20px'>{', '.join(top_words)}</span>", unsafe_allow_html=True)

                # Summary
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

# --- Info ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.markdown("""---""")
with st.expander("Tentang Aplikasi ℹ️", expanded=False):
    st.markdown(
        """
        Aplikasi ini digunakan untuk mendeteksi apakah sebuah artikel mengandung informasi hoaks atau tidak berdasarkan teks yang dimasukkan.
        """
    )

# --- Footer ---
st.markdown(
    """
    <hr style='border-top: 1px solid #bbb;'>
    <div style='text-align: center; font-size: 14px;'>
        Dibuat oleh <b>Mohammad Ramadhan Zainoor</b> · © 2025 · <a href="https://github.com/zainoor" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)
