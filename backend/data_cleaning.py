import pandas as pd

# Load dataset
df = pd.read_csv("data/reddit_posts.csv")

print("Original Shape:")
print(df.shape)

# Select useful columns
df = df[[
    "post_title",
    "subreddit",
    "score",
    "comments",
    "upvote_ratio",
    "year"
]]

print("\nNew Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# Save cleaned dataset
df.to_csv("data/cleaned_reddit_posts.csv", index=False)

print("\nDataset Saved Successfully!")