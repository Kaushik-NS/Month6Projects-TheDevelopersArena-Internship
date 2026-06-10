import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv(
    "data/processed/model_data.csv"
)

print("========== DATA TYPES ==========")
print(df.dtypes)

# ====================================
# TARGET
# ====================================

y = df["price"]

# ====================================
# FEATURES
# ====================================

X = df.drop(
    columns=[
        "price",
        "date",
        "size",
        "url"
    ],
    errors="ignore"
)

# Keep numeric columns only
X = X.select_dtypes(
    include=["int64", "float64"]
)

print("\n========== FEATURE COLUMNS ==========")
print(X.columns.tolist())

print("\n========== FEATURE TYPES ==========")
print(X.dtypes)

# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# MODEL
# ====================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("\n========== TRAINING MODEL ==========")

model.fit(
    X_train,
    y_train
)

# ====================================
# PREDICTIONS
# ====================================

predictions = model.predict(
    X_test
)

# ====================================
# METRICS
# ====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

# ====================================
# RESULTS
# ====================================

print("\n========== MODEL RESULTS ==========")

print("MAE:", mae)

print("RMSE:", rmse)

print("R2 Score:", r2)

print("Accuracy Approximation:", round(r2 * 100, 2), "%")

print("\nSTEP 1 - Starting training")

model.fit(
    X_train,
    y_train
)

print("STEP 2 - Training completed")

predictions = model.predict(
    X_test
)

print("STEP 3 - Predictions completed")

mae = mean_absolute_error(
    y_test,
    predictions
)

print("STEP 4 - MAE calculated")

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print("STEP 5 - RMSE calculated")

r2 = r2_score(
    y_test,
    predictions
)

print("STEP 6 - R2 calculated")

print("\n========== MODEL RESULTS ==========")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ====================================
# SAVE MODEL
# ====================================

joblib.dump(
    model,
    "models/house_price_model.pkl"
)

print(
    "\nModel saved successfully!"
)