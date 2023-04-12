import streamlit as st
import joblib

# Load the saved model
dtr_model = joblib.load('property_predictor.joblib')

# Get the unique city_location values from the dataset
city_locations = df['city_location'].unique()

# Create a selectbox for city_location
city_location = st.selectbox('Select city and location:', city_locations)

# Get the bedrooms and baths values from the user
bedrooms = st.number_input('Enter number of bedrooms:', value=2)
baths = st.number_input('Enter number of baths:', value=2)

# Get the total square feet area from the user
sqft = st.number_input('Enter total square feet area:', value=1000)

# Call the predict_price function to get the predicted price
predicted_price = predict_price(city_location.split('_')[0], city_location.split('_')[1], sqft, bedrooms, baths)

# Display the predicted price to the user
st.write('Predicted price for {} with {} bedrooms and {} baths and {} square feet area is {} Lakhs.'.format(city_location, bedrooms, baths, sqft, int(predicted_price)))
