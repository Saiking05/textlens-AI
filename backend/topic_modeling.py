import pandas as pd
from ast import literal_eval

from gensim import corpora
from gensim.models import LdaModel

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/lemmatized_reddit_posts.csv")

# Convert string representation of list back to list
df["lemmas"] = df["lemmas"].apply(literal_eval)

# ==========================================
# Create Dictionary
# ==========================================

dictionary = corpora.Dictionary(df["lemmas"])

# ==========================================
# Create Corpus
# ==========================================

corpus = [dictionary.doc2bow(text) for text in df["lemmas"]]

# ==========================================
# Train LDA Model
# ==========================================

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,      # Number of topics
    random_state=42,
    passes=10,
    alpha="auto"
)

# ==========================================
# Print Topics
# ==========================================

print("=" * 70)
print("Top 5 Topics")
print("=" * 70)

topics = lda_model.print_topics(num_words=10)

for topic_no, topic in topics:
    print(f"\nTopic {topic_no + 1}")
    print(topic)

# ==========================================
# Save Topics
# ==========================================

topic_list = []

for topic_no, topic in topics:
    topic_list.append({
        "Topic": topic_no + 1,
        "Keywords": topic
    })

topic_df = pd.DataFrame(topic_list)

topic_df.to_csv("data/topic_modeling.csv", index=False)

print("\n✅ Topic Modeling Completed Successfully!")
print("📁 Saved : data/topic_modeling.csv")