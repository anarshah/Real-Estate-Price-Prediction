import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load preprocessed data and trained model
df = pd.read_csv('property_updated_csv.csv')
dtr = load('property_predictor.joblib')

# Get list of unique city_location values
city_locations = df['city_location'].unique()

# Function to predict price
def predict_price(city_location, sqft, bedrooms, baths):
    loc_index = np.where(city_locations == city_location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return dtr.predict([x])[0] / 100000

# Create X dataframe for prediction function
X = df.drop(['price', 'price_per_sqft'], axis=1)

# Streamlit app title
st.title("Property Price Predictor")

# User input fields
city_location = st.selectbox("City and Location", city_locations)
sqft = st.number_input("Area (in square feet)", value=1000)
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
baths = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(city_location, sqft, bedrooms, baths)
    st.write("The estimated price is {:.2f} Lakhs".format(predicted_price))
