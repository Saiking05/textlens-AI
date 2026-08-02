import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv("data/preprocessed_reddit_posts.csv")

# Create analyzer
analyzer = SentimentIntensityAnalyzer()

# Function
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))

    if score["compound"] >= 0.05:
        return "Positive"

    elif score["compound"] <= -0.05:
        return "Negative"

    else:
        return "Neutral"

# Apply
df["sentiment"] = df["clean_title"].apply(get_sentiment)

# Show sample
print("=" * 60)
print(df[["clean_title", "sentiment"]].head(10))

print("\nSentiment Counts")
print(df["sentiment"].value_counts())

# Save
df.to_csv("data/sentiment_reddit_posts.csv", index=False)

print("\n✅ Sentiment Analysis Completed!")