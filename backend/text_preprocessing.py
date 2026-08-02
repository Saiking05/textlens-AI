import pandas as pd
import re

#load cleaned dataset
df=pd.read_csv("data/cleaned_reddit_posts.csv")

#function to clean text
def clean_text(text):

    #convert to string
    text=str(text)

    #lowercase
    text=text.lower()

    #remove url
    text=re.sub(r"http\S+|www\S+","",text)

    #remove punctuation
    text=re.sub(r"[^\w\s]","",text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df["clean_title"] = df["post_title"].apply(clean_text)

# Show before and after
print("=" * 60)

print("Original:")
print(df["post_title"].iloc[0])

print("\nCleaned:")
print(df["clean_title"].iloc[0])

# Save dataset
df.to_csv("data/preprocessed_reddit_posts.csv", index=False)

print("\nPreprocessing Completed Successfully!")