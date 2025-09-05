import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("car_price_model.pkl")

st.title("Car Price Prediction App")
st.write("Enter car details below to predict its price.")

# Car name (just for display, not used in prediction)
car_name = st.text_input("Car Name")

# Inputs required by model
year = st.number_input("Year of Manufacture", 1990, 2025, 2018)
present_price = st.number_input("Present Price (in lakhs)", 0.0, 100.0, 5.0, step=0.5)
kms_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Encode categorical variables (must match your training preprocessing)
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_mapping = {"Dealer": 0, "Individual": 1}
transmission_mapping = {"Manual": 0, "Automatic": 1}

fuel_encoded = fuel_mapping[fuel_type]
seller_encoded = seller_mapping[seller_type]
transmission_encoded = transmission_mapping[transmission]

# Prepare features in same order as training dataset
features = np.array([[year, present_price, kms_driven,
                      fuel_encoded, seller_encoded,
                      transmission_encoded, owner]])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(features)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:,.2f} lakhs")
