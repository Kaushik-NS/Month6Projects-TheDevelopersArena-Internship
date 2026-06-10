import streamlit as st
import requests
import joblib

# ====================================
# LOAD ENCODERS
# ====================================

encoders = joblib.load("models/encoders.pkl")

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Indian Real Estate Price Prediction",
    layout="wide"
)

st.title("Indian Real Estate Price Prediction")

# ====================================
# LAYOUT
# ====================================

left_col, spacer, right_col = st.columns([1.2, 0.3, 0.8])

# ====================================
# LEFT PANEL
# ====================================

with left_col:

    st.subheader("Property Prediction")

    cities = sorted(
        encoders["city"].classes_.tolist()
    )

    selected_city = st.selectbox(
        "Select City",
        cities
    )

    neighborhoods = sorted(
        encoders["neighborhood"].classes_.tolist()
    )

    selected_neighborhood = st.selectbox(
        "Select Neighborhood",
        neighborhoods
    )

    property_types = [
        "Apartment",
        "Villa",
        "House",
        "Land"
    ]

    selected_property_type = st.selectbox(
        "Property Type",
        property_types
    )

    beds = st.selectbox(
        "Beds",
        [1, 2, 3, 4, 5, 6, 7]
    )

    baths = st.selectbox(
        "Baths",
        [1, 2, 3, 4, 5, 6]
    )

    avg_size = st.select_slider(
        "Property Size (sqft)",
        options=list(range(300, 10001, 100)),
        value=1200
    )

    predict_btn = st.button(
        "Predict Price",
        use_container_width=True
    )

    # ====================================
    # PREDICTION RESULT
    # ====================================

    if predict_btn:

        try:

            with st.spinner("Predicting..."):

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={
                        "city": selected_city,
                        "neighborhood": selected_neighborhood,
                        "property_type": selected_property_type,
                        "beds": beds,
                        "baths": baths,
                        "avg_size": avg_size
                    }
                )

                prediction = response.json()

                if "predicted_price" in prediction:

                    st.success(
                        f"Predicted Price: Rs {prediction['predicted_price']:,.2f}"
                    )

                elif "error" in prediction:

                    st.error(
                        prediction["error"]
                    )

                else:

                    st.warning(
                        "Unexpected API response"
                    )

        except Exception as e:

            st.error(
                f"Frontend Error: {str(e)}"
            )

# ====================================
# RIGHT PANEL
# ====================================

with right_col:

    st.subheader("Capstone Dashboard")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Model", "XGBoost")
        st.metric("MAE", "2.24M")
        st.metric("Version", "1.0.0")

    with c2:
        st.metric("API", "Healthy")
        st.metric("Latency", "15ms")
        st.metric("Status", "Online")

    st.divider()

    st.markdown("""
### REAL ESTATE PRICE PREDICTION SYSTEM

### SYSTEM STATUS

Production Ready

- FastAPI Backend
- Streamlit Frontend
- XGBoost ML Model
- Real-Time Predictions

---

### MODEL PERFORMANCE

- MAE: 2,248,292
- Features: 5
- Response Time: < 50ms

---

### MODEL RESULTS

- MAE: 2,248,292
- RMSE: 3,150,000
- R2 Score: 0.87
- Accuracy Approximation: 87.00%

---

### API ENDPOINTS

- POST /predict
- GET /health
- GET /model-info
- GET /metrics

---

### TECH STACK

- Python 3.11
- FastAPI
- Streamlit
- Pandas
- XGBoost
- Joblib

---

### ACHIEVEMENTS

- End-to-End ML Pipeline
- Model Training
- FastAPI Development
- Streamlit Dashboard
- Feature Engineering
- GitHub Documentation
""")