import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load frequency file
df = pd.read_csv("data/word_frequency.csv")

# Convert dataframe into dictionary
word_dict = dict(zip(df["Word"], df["Frequency"]))

# Generate word cloud
wc = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate_from_frequencies(word_dict)

# Show
plt.figure(figsize=(15,7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Reddit NLP Word Cloud", fontsize=20)
plt.show()

# Save image
wc.to_file("outputs/wordcloud.png")

print("✅ Word Cloud Saved!")