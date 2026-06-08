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
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Indian Real Estate Price Prediction")

# ====================================
# CITY
# ====================================

cities = sorted(
    encoders["city"].classes_.tolist()
)

selected_city = st.selectbox(
    "Select City",
    cities
)

# ====================================
# NEIGHBORHOOD
# ====================================

neighborhoods = sorted(
    encoders["neighborhood"].classes_.tolist()
)

selected_neighborhood = st.selectbox(
    "Select Neighborhood",
    neighborhoods
)

# ====================================
# PROPERTY TYPE
# ====================================

property_types = [
    "Apartment",
    "Villa",
    "House",
    "Land",
    "Builder Floor",
    "Plot"
]

selected_property_type = st.selectbox(
    "Property Type",
    property_types
)

# ====================================
# BEDS
# ====================================

beds = st.selectbox(
    "Beds",
    [1, 2, 3, 4, 5, 6, 7]
)

# ====================================
# BATHS
# ====================================

baths = st.selectbox(
    "Baths",
    [1, 2, 3, 4, 5, 6]
)

# ====================================
# SIZE
# ====================================

avg_size = st.select_slider(
    "Property Size (sqft)",
    options=list(range(300, 10001, 100)),
    value=1200
)

# ====================================
# PREDICT
# ====================================

if st.button("Predict Price"):

    try:

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

        st.write(prediction)

        if "predicted_price" in prediction:

            st.success(
                f"🏷️ Predicted Price: ₹ {prediction['predicted_price']:,.2f}"
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