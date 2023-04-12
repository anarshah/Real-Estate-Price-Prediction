import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
dtr = load('property_predictor.joblib')

# Load preprocessed data
df = pd.read_csv('property_updated_csv.csv')

# Define the function to make predictions
def predict_price(city, location, sqft, bedrooms, baths):
    city_location = city + '_' + location
    X = pd.get_dummies(df.drop("price", axis=1))

    if city_location in X.columns:
        loc_index = np.where(X.columns == city_location)[0][0]
    else:
        loc_index = -1

    x = np.zeros(len(X.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return dtr.predict([x])[0] / 100000

# Define Streamlit app
st.title('Real Estate Price Prediction')

city = st.selectbox('City', df['city'].unique())
location = st.selectbox('Location', df['location'].unique())
sqft = st.number_input('Square Feet', min_value=300, step=10)
bedrooms = st.number_input('Bedrooms', min_value=1, step=1)
baths = st.number_input('Bathrooms', min_value=1, step=1)

if st.button('Predict Price'):
    predicted_price = predict_price(city, location, sqft, bedrooms, baths)
    st.write(f"Predicted Price: PKR {predicted_price} Lakhs")
