from fastapi import FastAPI
from enum import Enum
import pandas as pd
import joblib
import logging
import time

from backend.config import (
    MODEL_PATH,
    ENCODER_PATH,
    MODEL_VERSION
)

from backend.api.schemas import PredictionRequest

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
# ENUMS
# ====================================

class CityEnum(str, Enum):
    Chennai = "Chennai"
    Bangalore = "Bangalore"
    Hyderabad = "Hyderabad"
    Mumbai = "Mumbai"
    Delhi = "Delhi"


class NeighborhoodEnum(str, Enum):
    Adyar = "Adyar"
    Velachery = "Velachery"
    Anna_Nagar = "Anna Nagar"
    T_Nagar = "T Nagar"


class TypeEnum(str, Enum):
    Apartment = "Apartment"
    Villa = "Villa"
    House = "House"

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

    city = request.city
    neighborhood = request.neighborhood
    property_type = request.property_type

    beds = request.beds
    baths = request.baths
    avg_size = request.avg_size

    # existing prediction code

    start_time = time.time()

    try:

        city_encoded = encoders["city"].transform(
            [city.value]
        )[0]

        neighborhood_encoded = encoders[
            "neighborhood"
        ].transform(
            [neighborhood.value]
        )[0]

        type_encoded = encoders["type"].transform(
            [type.value]
        )[0]

        total_rooms = beds + baths

        data = pd.DataFrame([{
            "beds": beds,
            "baths": baths,
            "city": city_encoded,
            "neighborhood": neighborhood_encoded,
            "type": type_encoded,
            "avg_size": avg_size,
            "total_rooms": total_rooms
        }])

        prediction = model.predict(data)[0]

        response_time = round(
            time.time() - start_time,
            4
        )

        logging.info(
            f"Prediction successful | "
            f"City={city.value} | "
            f"Prediction={prediction}"
        )

        return {
            "predicted_price": float(prediction),
            "response_time_seconds": response_time
        }

    except Exception as e:

        logging.error(
            f"Prediction failed: {str(e)}"
        )

        return {
            "error": str(e)
        }