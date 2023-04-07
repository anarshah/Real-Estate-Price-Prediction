import streamlit as st
import joblib

# Load the trained decision tree model
decision_tree_model = joblib.load('decision_tree_model.joblib')

# Load the LabelEncoder
le = joblib.load('label_encoder.joblib')

# Create a list of available locations
locations = ['Bahria Town', 'DHA Defence', 'Gulberg', 'Johar Town', 'Wapda Town']

# Get the user input
st.title("Real Estate Price Predictor")
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Square Feet")
bedrooms = st.number_input("Enter Number of Bedrooms")
bathrooms = st.number_input("Enter Number of Bathrooms")
 
# Transform location using the LabelEncoder
le.transform(locations)

# Make the prediction
location_encoded = le.transform([location])[0]
prediction = decision_tree_model.predict([[location_encoded, sqft, bedrooms, bathrooms]])

# Display the prediction
st.write("The estimated price of the property is", prediction[0])
