import pandas as pd

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/lemmatized_reddit_posts.csv")

# Replace NaN
df["clean_title"] = df["clean_title"].fillna("").astype(str)

# Create summarizer
summarizer = LsaSummarizer()

# ==========================================
# Function
# ==========================================

def summarize_text(text):

    if len(text.split()) < 8:
        return text

    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    summary = summarizer(parser.document, 1)

    summary = " ".join(str(sentence) for sentence in summary)

    if summary == "":
        return text

    return summary

# ==========================================
# Apply
# ==========================================

df["summary"] = df["clean_title"].apply(summarize_text)

# ==========================================
# Show Sample
# ==========================================

print("=" * 70)

for i in range(10):

    print(f"\nOriginal : {df['clean_title'].iloc[i]}")
    print(f"Summary : {df['summary'].iloc[i]}")

print("=" * 70)

# ==========================================
# Save
# ==========================================

df.to_csv("data/summary_reddit_posts.csv", index=False)

print("\n✅ Text Summarization Completed Successfully!")
print("📁 Saved : data/summary_reddit_posts.csv")