# preprocess_all.py

import pandas as pd
import os
import re
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup
tqdm.pandas()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return stemmer.stem(text)

def clean_excel_files():
    files = {
        "cnn": "dataset/cnn.xlsx",
        "kompas": "dataset/kompas.xlsx",
        "tempo": "dataset/tempo.xlsx",
        "turnbackhoax": "dataset/turnbackhoax.xlsx"
    }
    cleaned_dfs = []

    print("📦 Membersihkan data Excel...")

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ File tidak ditemukan: {path}")
            continue

        df = pd.read_excel(path)

        # Pilih kolom teks
        if "text_new" in df.columns and "hoax" in df.columns:
            text_col = "text_new"
        elif "Narasi" in df.columns and "hoax" in df.columns:
            text_col = "Narasi"
        elif "FullText" in df.columns and "hoax" in df.columns:
            text_col = "FullText"
        else:
            print(f"⚠️ Kolom tidak dikenali di {path}")
            continue

        df = df[[text_col, "hoax"]].dropna()
        df.columns = ["text", "label"]
        df["cleaned"] = df["text"].progress_apply(preprocess)

        # Save cleaned version
        output_file = f"cleandataset/{name}_cleaned.csv"
        os.makedirs("cleandataset", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ Disimpan: {output_file}")

        cleaned_dfs.append(df[["cleaned", "label"]])

    return cleaned_dfs

def load_old_cleaned():
    df1 = pd.read_csv("cleandataset/politik_cleaned.csv")
    df2 = pd.read_csv("cleandataset/hoaxvalid_cleaned.csv")

    df1 = df1.rename(columns={'cleaned_text': 'cleaned'}) if 'cleaned_text' in df1.columns else df1
    df2 = df2.rename(columns={'cleaned_text': 'cleaned'}) if 'cleaned_text' in df2.columns else df2

    df1 = df1[['cleaned', 'label']].dropna()
    df2 = df2[['cleaned', 'label']].dropna()

    return pd.concat([df1, df2], ignore_index=True)

def main():
    print("🚀 Memulai proses preprocessing seluruh data...\n")
    old_data = load_old_cleaned()
    new_data = pd.concat(clean_excel_files(), ignore_index=True)

    final_df = pd.concat([old_data, new_data], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_output = "cleandataset/hoax_dataset_merged_v2.csv"
    final_df.to_csv(final_output, index=False)

    print("\n🎉 Semua proses selesai!")
    print(f"📁 Dataset akhir disimpan di: {final_output}")
    print("🔢 Total data:", len(final_df))
    print("📊 Distribusi label:\n", final_df['label'].value_counts())

if __name__ == "__main__":
    main()
