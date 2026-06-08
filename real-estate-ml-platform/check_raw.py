import pandas as pd

df = pd.read_csv("data/raw/real_estate_dataset.csv")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())