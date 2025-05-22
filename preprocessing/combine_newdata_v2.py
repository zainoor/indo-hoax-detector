import pandas as pd
import os
import re
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Folder dan file
old_file = "cleandataset/hoax_dataset_merged.csv"
new_files = [
    "dataset/cnn.xlsx",
    "dataset/kompas.xlsx",
    "dataset/tempo.xlsx",
    "dataset/turnbackhoax.xlsx"
]

# Load dataset lama
merged_old = pd.read_csv(old_file)

# Helper: Preprocessing + Stemming
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove links
    text = re.sub(r"[^a-z\s]", " ", text)             # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()
    return stemmer.stem(text)

# Baca dan bersihkan semua data baru
cleaned_rows = []

print("📦 Membersihkan data baru...")

for file in tqdm(new_files, desc="Proses file baru"):
    df = pd.read_excel(file)

    # Tentukan kolom teks & label
    if "text_new" in df.columns and "hoax" in df.columns:
        text_col = "text_new"
        label_col = "hoax"
    elif "Narasi" in df.columns and "hoax" in df.columns:
        text_col = "Narasi"
        label_col = "hoax"
    elif "FullText" in df.columns and "hoax" in df.columns:
        text_col = "FullText"
        label_col = "hoax"
    else:
        continue

    # Bersihkan data kosong
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Stemming + cleaning dengan tqdm
    tqdm.pandas(desc=" - Membersihkan teks")
    df["cleaned"] = df["text"].progress_apply(preprocess)

    # Simpan hasil bersih
    cleaned_rows.append(df[["cleaned", "label"]])

# Gabungkan semua data baru
new_combined = pd.concat(cleaned_rows, ignore_index=True)

# Gabungkan dengan data lama
final_combined = pd.concat([merged_old[["cleaned", "label"]], new_combined], ignore_index=True)

# Simpan
os.makedirs("cleandataset", exist_ok=True)
final_combined.to_csv("cleandataset/hoax_dataset_merged_v2.csv", index=False)

print("✅ Data akhir berhasil disimpan sebagai: cleandataset/hoax_dataset_merged_v2.csv")
