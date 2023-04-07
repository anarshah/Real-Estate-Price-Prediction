import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.joblib')

# Define the input fields
st.sidebar.header('Property Information')
property_type = st.sidebar.selectbox('Property Type', ['House', 'Apartment', 'Condo'])
location = st.sidebar.text_input('Location')
city = st.sidebar.selectbox('City', ['Toronto', 'Vancouver', 'Montreal'])
province_name = st.sidebar.selectbox('Province', ['Ontario', 'British Columbia', 'Quebec'])
latitude = st.sidebar.number_input('Latitude')
longitude = st.sidebar.number_input('Longitude')
baths = st.sidebar.number_input('Number of Bathrooms', value=1)
area = st.sidebar.number_input('Area (in sqft)', value=500)
bedrooms = st.sidebar.number_input('Number of Bedrooms', value=1)

# Create a DataFrame with the user input
input_df = pd.DataFrame({
    'property_type': property_type,
    'location': location,
    'city': city,
    'province_name': province_name,
    'latitude': latitude,
    'longitude': longitude,
    'baths': baths,
    'area': area,
    'bedrooms': bedrooms
}, index=[0])

# Make a prediction using the input DataFrame
prediction = model.predict(input_df)[0]

# Show the prediction
st.write('The predicted price of this property is $', prediction)
