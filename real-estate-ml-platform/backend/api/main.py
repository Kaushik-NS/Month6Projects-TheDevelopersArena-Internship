from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import time
import os

from backend.config import (
    MODEL_PATH,
    ENCODER_PATH,
    MODEL_VERSION
)

# ====================================
# CREATE LOGS FOLDER
# ====================================

os.makedirs("logs", exist_ok=True)

# ====================================
# LOGGING
# ====================================

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================
# FASTAPI APP
# ====================================

app = FastAPI(
    title="Indian Real Estate Prediction API",
    description="Predict Indian property prices using XGBoost",
    version=MODEL_VERSION
)

# ====================================
# LOAD MODEL & ENCODERS
# ====================================

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# ====================================
# REQUEST MODEL
# ====================================

class PredictionRequest(BaseModel):
    city: str
    neighborhood: str
    property_type: str
    beds: int
    baths: int
    avg_size: float

# ====================================
# ROOT
# ====================================

@app.get("/")
def home():
    return {
        "message": "Indian Real Estate Prediction API Running"
    }

# ====================================
# HEALTH CHECK
# ====================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ====================================
# MODEL INFO
# ====================================

@app.get("/model-info")
def model_info():
    return {
        "model_name": "Indian Real Estate Price Predictor",
        "algorithm": "XGBoost",
        "version": MODEL_VERSION
    }

# ====================================
# METRICS
# ====================================

@app.get("/metrics")
def metrics():
    return {
        "status": "running",
        "model_version": MODEL_VERSION
    }

# ====================================
# PREDICTION
# ====================================

@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    try:

        city = request.city.strip()
        neighborhood = request.neighborhood.strip()
        property_type = request.property_type.strip()

        beds = int(request.beds)
        baths = int(request.baths)
        avg_size = float(request.avg_size)

        print("\n========== REQUEST ==========")
        print("CITY =", city)
        print("NEIGHBORHOOD =", neighborhood)
        print("TYPE =", property_type)

        city_encoded = encoders["city"].transform(
            [city]
        )[0]

        neighborhood_encoded = encoders[
            "neighborhood"
        ].transform(
            [neighborhood]
        )[0]

        type_encoded = encoders["type"].transform(
        [property_type]
)[0]


        data = pd.DataFrame([{
    "beds": beds,
    "city": city_encoded,
    "type": type_encoded,
    "baths": baths,
    "neighborhood": neighborhood_encoded
}])

        print("\n========== MODEL INPUT ==========")
        print(data)

        print("\n========== DTYPES ==========")
        print(data.dtypes)

        prediction = model.predict(data)[0]

        print("\n===== MODEL INPUT =====")
        print(data)

        print("\n===== DTYPES =====")
        print(data.dtypes)

        print("\n===== VALUES =====")
        print(data.iloc[0].to_dict())

        response_time = round(
            time.time() - start_time,
            4
        )

        logging.info(
            f"Prediction successful | "
            f"City={city} | "
            f"Prediction={prediction}"
        )

        return {
            "predicted_price": float(prediction),
            "response_time_seconds": response_time
        }

    except Exception as e:

        print("\n========== ERROR ==========")
        print(str(e))

        logging.error(
            f"Prediction failed: {str(e)}"
        )

        return {
            "error": str(e)
        }