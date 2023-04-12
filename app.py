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
df1 = df.drop("price_per_sqft", axis=1)
dummies = pd.get_dummies(df1['city_location'])
dummies.head(3)
df1 = pd.concat([df1, dummies.drop('others', axis=1)], axis="columns")
df1 = df1.drop("city_location", axis=1)

X = df1.drop('price', axis=1) # Features
y = df1['price'] # Predictor or predicted_variable


def predict_price(city, location, sqft, bedrooms, baths):
    city_location = city + '_' + location
    if city_location not in data['city_location'].unique():
        print(f"City location '{city_location}' not found in the dataset.")
        return None

    x = np.zeros(len(X.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms

    loc_index = np.where(X.columns == city_location)[0][0]
    if loc_index >= 0:
        x[loc_index] = 1

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
