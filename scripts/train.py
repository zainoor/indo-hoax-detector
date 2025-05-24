import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleandataset/hoax_dataset_merged_v2.csv")
df = df.dropna(subset=["cleaned", "label"])

# Undersample to balance classes
hoax_df = df[df["label"] == 1]
valid_df = df[df["label"] == 0].sample(n=len(hoax_df), random_state=42)
df = pd.concat([hoax_df, valid_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X_text = df["cleaned"]
y = df["label"]

print("🔁 Performing 5-fold cross-validation...")

vectorizer_cv = TfidfVectorizer()
X_full = vectorizer_cv.fit_transform(X_text)
model_cv = LogisticRegression()

cv_results = cross_validate(model_cv, X_full, y, cv=5,
                            scoring=['accuracy', 'precision', 'recall', 'f1'])

print("\n📊 Cross-Validation Metrics:")
for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    scores = cv_results[metric]
    print(f"{metric}: {scores} | Mean: {np.mean(scores):.4f}")

print("\n🚀 Training final model on train/test split...")

X_text_train, X_text_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_text_train)
X_test = vectorizer.transform(X_text_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Apply threshold
threshold = 0.7
y_pred_thresh = (y_proba > threshold).astype(int)

# Save everything for visualization
joblib.dump(model, "models/hoax_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump((X_text_test, y_test, y_pred, y_proba, y_pred_thresh), "models/test_data.pkl")

