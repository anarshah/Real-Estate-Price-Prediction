import streamlit as st
import joblib
import pandas as pd

# Load the label encoder and decision tree model
le = joblib.load('label_encoder.joblib')
model = joblib.load('decision_tree_model.joblib')

# Create a function to preprocess user input and make predictions
def predict_price(location, area, bedrooms, bathrooms):
    # Load the data and encode the location column
    data = pd.DataFrame({
        'location': [location],
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })
    data['location'] = le.transform(data['location'])
    
    # Make a prediction using the model
    price = model.predict(data)[0]
    return price

# Create a Streamlit app
st.title("Property Price Predictor")

# Create input fields for user input
location = st.selectbox("Location", options=df['location'].unique())
area = st.number_input("Area (in Marla)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

# Create a button to make the prediction
if st.button("Predict Price"):
    price = predict_price(location, area, bedrooms, bathrooms)
    st.write("The predicted price is:", price)
