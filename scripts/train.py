import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleandataset/hoax_dataset_merged_v2.csv")

# Drop rows with NaN in 'cleaned' or 'label'
df = df.dropna(subset=["cleaned", "label"])

# Undersampling
# Separate classes
hoax_df = df[df["label"] == 1]
valid_df = df[df["label"] == 0].sample(n=len(hoax_df), random_state=42)

# Combine and shuffle
df = pd.concat([hoax_df, valid_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split first, keeping the original text
X_text = df["cleaned"]
y = df["label"]

X_text_train, X_text_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_text_train)
X_test = vectorizer.transform(X_text_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Save everything
joblib.dump(model, "models/hoax_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump((X_text_test, y_test, y_pred), "models/test_data.pkl")

