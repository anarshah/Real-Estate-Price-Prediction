import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load the model
model = joblib.load("zameen_property_model.joblib")

# Create a function to get user inputs
def get_user_inputs():
    property_type = st.selectbox("Select Property Type", ["House", "Apartment", "Land", "Farm House", "Penthouse"])
    location = st.text_input("Enter Location")
    city = st.text_input("Enter City")
    province_name = st.text_input("Enter Province Name")
    bedrooms = st.number_input("Enter number of bedrooms", min_value=0, max_value=20, value=1)
    bathrooms = st.number_input("Enter number of bathrooms", min_value=0, max_value=20, value=1)
    area = st.number_input("Enter area in square feet", min_value=0, max_value=10000000, value=1000)
    purpose = st.selectbox("Select Purpose", ["For Sale", "For Rent"])
    return pd.DataFrame({
        "property_type": [property_type],
        "location": [location],
        "city": [city],
        "province_name": [province_name],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "area": [area],
        "purpose": [purpose],
    })

# Create a function to make predictions
def make_predictions(model, inputs):
    inputs_encoded = inputs.apply(LabelEncoder().fit_transform)
    return model.predict(inputs_encoded)

# Create the Streamlit app
st.title("Zameen Property Price Predictor")
st.write("Enter the details of the property to get an estimated price.")

inputs = get_user_inputs()
if st.button("Predict Price"):
    price = make_predictions(model, inputs)
    st.write("The estimated price of the property is: ", price)
