import pandas as pd

# Load datasets
df1 = pd.read_csv("cleandataset/politik_cleaned.csv")          
df2 = pd.read_csv("cleandataset/hoaxvalid_cleaned.csv")       

# Standardize column names if needed
# Let's make sure both have 'cleaned' and 'label'
df1 = df1.rename(columns={'cleaned_text': 'cleaned'}) if 'cleaned_text' in df1.columns else df1
df2 = df2.rename(columns={'cleaned_text': 'cleaned'}) if 'cleaned_text' in df2.columns else df2

# Drop any nulls just in case
df1 = df1[['cleaned', 'label']].dropna()
df2 = df2[['cleaned', 'label']].dropna()

# Merge datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Shuffle to mix both sources
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new file
merged_df.to_csv("cleandataset/hoax_dataset_merged.csv", index=False)

print("Merged dataset saved as hoax_dataset_merged.csv")
print("Total samples:", len(merged_df))
print("Label distribution:\n", merged_df['label'].value_counts())