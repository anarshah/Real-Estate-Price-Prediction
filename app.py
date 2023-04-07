import streamlit as st
import joblib

# Load the saved model
model = joblib.load('decision_tree_model.joblib')

# Create a function to take input and make predictions
def predict_price(feature_vector):
    # Reshape the feature vector to match the expected input shape of the model
    feature_vector = feature_vector.reshape(1, -1)
    # Make a prediction using the loaded model
    prediction = model.predict(feature_vector)
    return prediction[0]

# Create a Streamlit app
def app():
    # Create a form to take input from the user
    st.write("Enter the features of the property:")
    area = st.number_input("Area (in square feet)")
    bedrooms = st.number_input("Number of bedrooms")
    bathrooms = st.number_input("Number of bathrooms")
    location = st.text_input("Location")
    # Convert the user input to a feature vector
    feature_vector = [area, bedrooms, bathrooms, location]
    # Make a prediction using the loaded model and the user input
    prediction = predict_price(feature_vector)
    # Display the prediction to the user
    st.write("The estimated price of the property is:", prediction)
