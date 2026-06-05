import pandas as pd
import re

# Load ORIGINAL raw dataset
df = pd.read_csv("data/raw/real_estate_dataset.csv")

# =========================
# Extract average size
# =========================

def extract_avg_size(size):

    nums = re.findall(r'\d+', str(size))
    nums = [int(n) for n in nums]

    if len(nums) == 2:
        return sum(nums) / 2

    elif len(nums) == 1:
        return nums[0]

    return None


df["avg_size"] = df["size"].apply(extract_avg_size)

# =========================
# Total rooms
# =========================

df["total_rooms"] = df["beds"] + df["baths"]

# =========================
# Drop URL ONLY
# =========================

if "url" in df.columns:
    df.drop(columns=["url"], inplace=True)

# =========================
# SAVE CLEAN HUMAN DATA
# =========================

df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print(df.head())

print("cleaned_data.csv created successfully")