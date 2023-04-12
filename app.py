import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('property_predictor.joblib')

# Load the dataset and preprocess it as required
data = pd.read_csv('property_updated_csv.csv')
df = data.copy()

df = df.reset_index()
df = df.drop("index",axis=1)
# preprocessing code ...

# Define the input features for the model
X = df.drop('price', axis=1)

# Define the user input
city = st.selectbox('City', df['city'].unique())
location = st.selectbox('Location', df[df['city'] == city]['location'].unique())
sqft = st.number_input('Area in Square Feet')
bedrooms = st.slider('Bedrooms', 1, 10)
baths = st.slider('Bathrooms', 1, 10)

# Concatenate city and location to form the city_location string
city_location = city + '_' + location

# Find the index of the corresponding feature in X
loc_index = X.columns.get_loc(city_location)

# Define the input values for the model
x = [baths, sqft, bedrooms] + [0] * (len(X.columns) - 3)
if loc_index >= 0:
    x[loc_index] = 1

# Make a prediction using the trained model
price = model.predict([x])[0] / 100000

# Display the predicted price to the user
st.write(f"The predicted price for a {bedrooms} bedroom property with {baths} bathrooms and {sqft} square feet area in {location} is {price:.2f} lakhs.")
