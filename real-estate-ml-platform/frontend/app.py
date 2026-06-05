import streamlit as st
import requests

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
# CITY MAPPING
# ====================================

city_map = {
    0: "Chennai",
    1: "Bangalore",
    2: "Hyderabad",
    3: "Mumbai",
    4: "Delhi"
}

selected_city_name = st.selectbox(
    "Select City",
    list(city_map.values())
)

selected_city = list(city_map.keys())[
    list(city_map.values()).index(selected_city_name)
]

# ====================================
# NEIGHBORHOOD MAPPING
# ====================================

# ====================================
# CITY-BASED NEIGHBORHOODS
# ====================================

city_neighborhoods = {

    "Chennai": {
        55: "Adyar",
        27: "Velachery",
        29: "Anna Nagar",
        53: "T Nagar"
    },

    "Bangalore": {
        86: "Indiranagar",
        101: "Whitefield",
        102: "Koramangala"
    },

    "Hyderabad": {
        171: "Banjara Hills",
        173: "Jubilee Hills",
        174: "Gachibowli"
    },

    "Mumbai": {
        191: "Andheri",
        192: "Bandra",
        193: "Powai"
    },

    "Delhi": {
        210: "Saket",
        211: "Dwarka",
        212: "Rohini"
    }
}

# Get neighborhoods for selected city
selected_city_neighborhoods = city_neighborhoods[
    selected_city_name
]

selected_neighborhood_name = st.selectbox(
    "Select Neighborhood",
    list(selected_city_neighborhoods.values())
)

selected_neighborhood = list(
    selected_city_neighborhoods.keys()
)[
    list(selected_city_neighborhoods.values()).index(
        selected_neighborhood_name
    )
]

# ====================================
# PROPERTY TYPE
# ====================================

type_map = {
    0: "Apartment",
    1: "Villa",
    2: "House"
}

selected_type_name = st.selectbox(
    "Property Type",
    list(type_map.values())
)

selected_type = list(type_map.keys())[
    list(type_map.values()).index(
        selected_type_name
    )
]

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
# PREDICT BUTTON
# ====================================

if st.button("Predict Price"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={
            "beds": beds,
            "baths": baths,
            "city": selected_city,
            "neighborhood": selected_neighborhood,
            "type": selected_type,
            "avg_size": avg_size
        }
    )

    prediction = response.json()

    st.success(
        f"🏷️ Predicted Price: ₹ {prediction['predicted_price']:,.2f}"
    )