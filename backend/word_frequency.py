import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("data/lemmatized_reddit_posts.csv")

# Convert string to list
df["lemmas"] = df["lemmas"].apply(eval)

# Merge all words
all_words = []

for words in df["lemmas"]:
    all_words.extend(words)

# Count frequency
word_freq = Counter(all_words)

# Top 20 words
top_words = word_freq.most_common(20)

print("=" * 60)
print("Top 20 Most Frequent Words")
print("=" * 60)

for word, count in top_words:
    print(f"{word:<20} {count}")

# Save to CSV
freq_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
freq_df.to_csv("data/word_frequency.csv", index=False)

print("\n✅ Word Frequency Saved Successfully!")