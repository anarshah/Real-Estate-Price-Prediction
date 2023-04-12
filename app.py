import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model
dtr = load('property_predictor.joblib')

# Load the city and location data
city_locations = pd.read_csv('zameen-property-data.csv')['city_location'].unique()

# Create a Streamlit app
st.title('Property Price Predictor')

# Add user input options
city = st.selectbox('Select city:', np.unique([x.split('_')[0] for x in city_locations]))
locations = [x.split('_')[1] for x in city_locations if x.split('_')[0] == city]
location = st.selectbox('Select location:', locations)
sqft = st.slider('Enter the total square feet area:', 100, 5000, 500)
bedrooms = st.slider('Enter the number of bedrooms:', 1, 10, 2)
baths = st.slider('Enter the number of bathrooms:', 1, 10, 2)

# Define a function to make predictions
def predict_price(city, location, sqft, bedrooms, baths):
    city_location = city + '_' + location
    loc_index = np.where(X.columns == city_location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return dtr.predict([x])[0] / 100000

# Make predictions based on user input
city_location_value = city + '_' + location
predicted_price = predict_price(city, location, sqft, bedrooms, baths)

# Show the predicted price
st.subheader('Predicted Price:')
st.write(f'{int(predicted_price)} Lakhs')
