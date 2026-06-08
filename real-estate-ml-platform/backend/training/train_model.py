import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load encoded dataset
df = pd.read_csv("data/processed/model_data.csv")

print("========== DATA TYPES ==========")
print(df.dtypes)

# Target column
y = df["price"]

# Remove columns that should not be used for training
X = df.drop(
    columns=[
        "price",
        "date",
        "size",
        "url"
    ],
    errors="ignore"
)

# Keep only numeric columns
X = X.select_dtypes(
    include=["int64", "float64"]
)

print("\n========== FEATURE COLUMNS ==========")
print(X.columns.tolist())

print("\n========== FEATURE TYPES ==========")
print(X.dtypes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(df.columns.tolist())

# Train model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("\nALL COLUMNS:")
print(df.columns.tolist())

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(
    y_test,
    predictions
)

print("\n========== RESULTS ==========")
print("MAE:", mae)

# Save model
joblib.dump(
    model,
    "models/house_price_model.pkl"
)

print("\nModel saved successfully!")