import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load HUMAN READABLE dataset
df = pd.read_csv("data/raw/real_estate_dataset.csv")

# Create copy for encoding
encoded_df = df.copy()

encoders = {}

categorical_cols = ["city", "neighborhood", "type"]

for col in categorical_cols:

    le = LabelEncoder()

    encoded_df[col] = le.fit_transform(
        encoded_df[col].astype(str)
    )

    encoders[col] = le

    print(f"\n{col.upper()} CLASSES:")
    print(le.classes_[:20])

# Create folders if missing
os.makedirs("models", exist_ok=True)

# Save encoders
joblib.dump(
    encoders,
    "models/encoders.pkl"
)

# Save encoded dataset for ML training
encoded_df.to_csv(
    "data/processed/model_data.csv",
    index=False
)

print("\nFeature engineering complete")