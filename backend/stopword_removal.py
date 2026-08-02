import pandas as pd
import nltk
import ast
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("data/tokenized_reddit_posts.csv")

# Convert string back to list
df["tokens"] = df["tokens"].apply(ast.literal_eval)

# English stopwords
stop_words = set(stopwords.words("english"))#it gives ready made list of words like the, a, it etc.

# Remove stopwords
df["filtered_tokens"] = df["tokens"].apply(
    lambda words: [word for word in words if word.lower() not in stop_words]
)

# Show sample
print("=" * 60)

print("Before:")
print(df["tokens"].iloc[0])

print("\nAfter:")
print(df["filtered_tokens"].iloc[0])

# Save
df.to_csv("data/stopwords_removed.csv", index=False)

print("\n✅ Stopword Removal Completed Successfully!")