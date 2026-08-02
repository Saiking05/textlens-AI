import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download required resources (first time only)
nltk.download("punkt")
nltk.download("punkt_tab")

# Load preprocessed dataset
df = pd.read_csv("data/preprocessed_reddit_posts.csv")

# Convert NaN values to empty string
df["clean_title"] = df["clean_title"].fillna("").astype(str)

# Tokenization
df["tokens"] = df["clean_title"].apply(word_tokenize)

# Show sample output
print("=" * 60)
print("Original Text:")
print(df["clean_title"].iloc[0])

print("\nTokens:")
print(df["tokens"].iloc[0])

# Save tokenized dataset
df.to_csv("data/tokenized_reddit_posts.csv", index=False)

print("\n✅ Tokenization Completed Successfully!")
print("\nSaved File : data/tokenized_reddit_posts.csv")