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
    x = pd.DataFrame(columns=X.columns)
    x.loc[0] = np.zeros(len(X.columns))
    x.loc[0, 'baths'] = baths
    x.loc[0, 'area'] = sqft
    x.loc[0, 'bedrooms'] = bedrooms
    if city_location in x.columns:
        x.loc[0, city_location] = 1
    return dtr.predict(x.values)[0] / 100000

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
