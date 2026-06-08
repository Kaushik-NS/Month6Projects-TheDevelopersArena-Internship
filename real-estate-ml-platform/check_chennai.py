import pandas as pd

df = pd.read_csv("data/raw/real_estate_dataset.csv")

print(
    sorted(
        df[df["city"] == "Chennai"]["neighborhood"]
        .dropna()
        .unique()
        .tolist()
    )
)