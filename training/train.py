import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleandataset/hoax_dataset_merged.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict labels for evaluation
y_pred = model.predict(X_test)

# Save model, vectorizer, and test results to 'models' folder
joblib.dump(model, "models/hoax_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump((X_test, y_test, y_pred), "models/test_data.pkl")
