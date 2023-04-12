import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.tree import DecisionTreeRegressor

# Load the saved model
dtr = load('property_predictor.joblib')

# Read the preprocessed data
data = pd.read_csv('property_updated_csv.csv')

# Prepare the feature matrix X and target variable y
X = data.drop(['price', 'price_per_sqft', 'city_location'], axis=1)
y = data['price']

def predict_price(city, location, sqft, bedrooms, baths):
    city_location = city + '_' + location
    if city_location not in data['city_location'].unique():
        print(f"City location '{city_location}' not found in the dataset.")
        return None

    x = np.zeros(len(X.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    
    city_location_indices = data[data['city_location'] == city_location].index
    X_city_location = X.loc[city_location_indices]
    X_city_location = X_city_location.reset_index(drop=True)

    return dtr.predict([x])[0]

st.title('Real Estate Price Prediction')

city = st.selectbox('City', sorted(data['city_location'].str.split('_').str[0].unique()))
location = st.selectbox('Location', sorted(data[data['city_location'].str.startswith(city)]['city_location'].str.split('_').str[1].unique()))
sqft = st.number_input('Area in Square Feet', min_value=1, max_value=int(data['area'].max()), step=1)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=int(data['bedrooms'].max()), step=1)
baths = st.number_input('Number of Baths', min_value=1, max_value=int(data['baths'].max()), step=1)

if st.button('Predict Price'):
    predicted_price = predict_price(city, location, sqft, bedrooms, baths)
    if predicted_price:
        st.success(f"Predicted price: {predicted_price:.2f} PKR")
    else:
        st.error("Unable to make a prediction for the given city and location.")
