import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sentiment_reddit_posts.csv")

# Count sentiments
counts = df["sentiment"].value_counts()

print("=" * 50)
print("Sentiment Distribution")
print("=" * 50)
print(counts)

# ---------- Pie Chart ----------
plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Reddit Sentiment Analysis")

plt.savefig("outputs/sentiment_pie_chart.png")

plt.show()

# ---------- Bar Chart ----------
plt.figure(figsize=(6,4))

plt.bar(counts.index, counts.values)

plt.title("Sentiment Count")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")

plt.savefig("outputs/sentiment_bar_chart.png")

plt.show()

print("\n✅ Charts Saved Successfully!")