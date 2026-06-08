# 🏠 Indian Real Estate Price Prediction Platform

A Machine Learning-powered web application that predicts residential property prices across major Indian cities using historical real estate data and an XGBoost regression model.

## Overview

The Indian Real Estate Price Prediction Platform provides intelligent property valuation by analyzing key property attributes such as:

* City
* Neighborhood
* Property Type
* Number of Bedrooms
* Number of Bathrooms
* Property Size

The platform combines a machine learning backend, FastAPI REST API, and Streamlit frontend to deliver real-time property price predictions.

## Features

### Machine Learning

* XGBoost Regression Model
* Automated Feature Engineering
* Label Encoding for Categorical Features
* Model Performance Evaluation using MAE

### Backend API

* FastAPI-based REST API
* Health Monitoring Endpoints
* Prediction Endpoint
* Logging and Error Handling

### Frontend

* Interactive Streamlit Dashboard
* Real-Time Property Price Prediction
* User-Friendly Interface
* Responsive Design

### Monitoring

* API Health Checks
* Request Logging
* Model Version Tracking

## Project Architecture

```text
real-estate-ml-platform/
│
├── backend/
│   ├── api/
│   ├── preprocessing/
│   ├── training/
│   └── config.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── house_price_model.pkl
│   └── encoders.pkl
│
├── docs/
├── monitoring/
├── tests/
├── logs/
└── requirements.txt
```

## Technology Stack

| Layer               | Technology  |
| ------------------- | ----------- |
| Frontend            | Streamlit   |
| Backend             | FastAPI     |
| Machine Learning    | XGBoost     |
| Data Processing     | Pandas      |
| Model Serialization | Joblib      |
| Language            | Python 3.11 |

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd real-estate-ml-platform
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Train Model

```bash
python backend/training/train_model.py
```

### Start API

```bash
uvicorn backend.api.main:app --reload
```

### Start Frontend

```bash
streamlit run frontend/app.py
```

## API Endpoints

### Health Check

```http
GET /health
```

### Model Information

```http
GET /model-info
```

### Prediction

```http
POST /predict
```

Request:

```json
{
  "city": "Chennai",
  "neighborhood": "Adyar",
  "property_type": "Apartment",
  "beds": 3,
  "baths": 3,
  "avg_size": 1200
}
```

Response:

```json
{
  "predicted_price": 20292612.0,
  "response_time_seconds": 0.02
}
```

## Model Performance

Current Model:

* Algorithm: XGBoost Regressor
* Metric: Mean Absolute Error (MAE)
* Version: 1.0.0

## Future Enhancements

* Advanced Property Analytics
* Interactive Visualizations
* Model Retraining Pipeline
* Docker Deployment
* Cloud Deployment
* Property Recommendation System

## License

This project is released under the MIT License.

## Author

Kaushik NS

Game Developer | Machine Learning Enthusiast | Software Engineer
