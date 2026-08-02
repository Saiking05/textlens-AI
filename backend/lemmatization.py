import pandas as pd
import spacy
import ast

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("data/stopwords_removed.csv")

# Convert string to list
df["filtered_tokens"] = df["filtered_tokens"].apply(ast.literal_eval)

# Lemmatization function
def lemmatize_words(words):
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc]

# Apply lemmatization
df["lemmas"] = df["filtered_tokens"].apply(lemmatize_words)

# Show sample
print("=" * 60)
print("Before:")
print(df["filtered_tokens"].iloc[0])

print("\nAfter:")
print(df["lemmas"].iloc[0])

# Save dataset
df.to_csv("data/lemmatized_reddit_posts.csv", index=False)

print("\n✅ Lemmatization Completed Successfully!")