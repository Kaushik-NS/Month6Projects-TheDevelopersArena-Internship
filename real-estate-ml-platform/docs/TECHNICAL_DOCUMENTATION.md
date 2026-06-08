# Technical Documentation

## System Overview

The Indian Real Estate Price Prediction Platform is designed using a three-layer architecture:

1. Data Processing Layer
2. Machine Learning Layer
3. Application Layer

---

# Data Pipeline

## Raw Dataset

Location:

```text
data/raw/real_estate_dataset.csv
```

Contains:

* Property listings
* City information
* Neighborhood information
* Property type
* Beds
* Baths
* Price

---

# Feature Engineering

Script:

```text
backend/preprocessing/features.py
```

Responsibilities:

* Label Encoding
* Dataset Cleaning
* Feature Selection
* Encoder Serialization

Generated Artifacts:

```text
models/encoders.pkl
data/processed/model_data.csv
```

---

# Model Training

Script:

```text
backend/training/train_model.py
```

Algorithm:

```text
XGBoost Regressor
```

Features:

```text
beds
city
type
baths
neighborhood
```

Target:

```text
price
```

Generated Model:

```text
models/house_price_model.pkl
```

---

# Backend API

Framework:

```text
FastAPI
```

Main File:

```text
backend/api/main.py
```

Endpoints:

## GET /

Returns service status.

## GET /health

Health monitoring endpoint.

## GET /model-info

Returns model metadata.

## POST /predict

Performs price prediction.

---

# Frontend

Framework:

```text
Streamlit
```

File:

```text
frontend/app.py
```

Responsibilities:

* User Input Collection
* API Integration
* Prediction Display
* Error Handling

---

# Logging

Directory:

```text
logs/
```

Log File:

```text
api.log
```

Tracked Events:

* Requests
* Prediction Results
* Errors
* Response Times

---

# Model Workflow

User Input

↓

Frontend Validation

↓

API Request

↓

Encoder Transformation

↓

Feature Vector Creation

↓

XGBoost Prediction

↓

JSON Response

↓

Frontend Display

---

# Security Considerations

* Input Validation
* Exception Handling
* Structured Logging
* API Isolation

---

# Performance

Average Prediction Latency:

```text
< 50 ms
```

Model Loading:

```text
Startup Initialization
```

Inference:

```text
Single Record Prediction
```

---

# Deployment Strategy

Development:

```bash
uvicorn backend.api.main:app --reload
streamlit run frontend/app.py
```

Production:

* Docker Containers
* Nginx Reverse Proxy
* Cloud Hosting
* CI/CD Pipeline
