import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


# Load the trained model
model = joblib.load('model.joblib')

# Define a function to preprocess the input data
def preprocess_input(baths, bedrooms, area, location, city, province_name):
    # Preprocess the area feature
    pattern = r'^([\d\.]+)'
    area_num = float(re.findall(pattern, area)[0])
    
    # Create a DataFrame with the preprocessed input data
    input_df = pd.DataFrame({
        'baths': [baths],
        'bedrooms': [bedrooms],
        'area_num': [area_num],
        'location': [location],
        'city': [city],
        'province_name': [province_name]
    })
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    # Fit and transform 'location' column, handle unseen labels with 'ignore'
    # Fit and transform categorical columns
    input_df['location'] = le.fit_transform(input_df['location'].astype(str)).astype(int)
    input_df['city'] = le.fit_transform(input_df['city'].astype(str)).astype(int)
    input_df['province_name'] = le.fit_transform(input_df['province_name'].astype(str)).astype(int)
    
    # Generate polynomial features
    poly = joblib.load('polynomial_features.joblib')
    input_df = pd.DataFrame(poly.transform(input_df), columns=poly.get_feature_names(['baths', 'bedrooms', 'area_num']))
    
    return input_df

# Define the Streamlit app
def app():
    st.title('Property Price Predictor')
    
    # Define the input fields
    baths = st.number_input('Number of bathrooms', value=2, min_value=1, max_value=10)
    bedrooms = st.number_input('Number of bedrooms', value=3, min_value=1, max_value=10)
    area = st.text_input('Area (in square feet)', value='1200')
    location = st.selectbox('Location', ['DHA Phase 1', 'Bahria Town', 'Gulberg', 'Model Town', 'Johar Town', 'Iqbal Town', 'Township'])
    city = st.selectbox('City', ['Lahore', 'Islamabad', 'Karachi', 'Rawalpindi', 'Faisalabad', 'Multan', 'Gujranwala', 'Peshawar'])
    province_name = st.selectbox('Province', ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Islamabad Capital Territory', 'Balochistan', 'Gilgit-Baltistan', 'Azad Jammu and Kashmir'])
    
    # Preprocess the input data
    input_df = preprocess_input(baths, bedrooms, area, location, city, province_name)
    
    # Make the prediction
    predicted_price = model.predict(input_df)[0]
    
    # Display the predicted price
    st.write('The predicted price of the property is:', predicted_price)


if __name__ == '__main__':
    app()
