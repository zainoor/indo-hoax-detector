import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure the 'models' folder exists
os.makedirs("models", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("cleandataset/hoax_dataset_merged.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer to 'models' folder
joblib.dump(model, "models/hoax_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
