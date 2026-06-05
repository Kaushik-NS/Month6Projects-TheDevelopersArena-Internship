import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned human-readable dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create copy for ML model
encoded_df = df.copy()

encoders = {}

categorical_cols = ["city", "neighborhood", "type"]

# Encode categorical columns
for col in categorical_cols:

    le = LabelEncoder()

    encoded_df[col] = le.fit_transform(encoded_df[col])

    encoders[col] = le

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")

# Save encoded dataset for ML training
encoded_df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("Feature engineering complete")