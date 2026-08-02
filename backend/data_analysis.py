import pandas as pd

# Load Dataset
df = pd.read_csv("data/reddit_posts.csv")

print("="*50)
print("First 5 Rows")
print("="*50)
print(df.head())

print("\n")

print("="*50)
print("Shape")
print("="*50)
print(df.shape)

print("\n")

print("="*50)
print("Columns")
print("="*50)
print(df.columns)

print("\n")

print("="*50)
print("Information")
print("="*50)
df.info()

print("\n")

print("="*50)
print("Missing Values")
print("="*50)
print(df.isnull().sum())