import streamlit as st
import joblib
import numpy as np

# Load the saved decision tree model
decision_tree_model = joblib.load('decision_tree_model.joblib')

# Create a form for user input
st.write('Please enter the following information:')
form = st.form(key='prediction_form')
beds = form.number_input('Number of bedrooms:', min_value=1, max_value=10, value=2)
baths = form.number_input('Number of bathrooms:', min_value=1, max_value=10, value=2)
area = form.number_input('Area (in square feet):', min_value=1, value=1000)
location = form.selectbox('Location:', ['Gulberg', 'DHA', 'Bahria Town', 'Model Town'])
submit_button = form.form_submit_button(label='Get Prediction')

# Make a prediction based on user input
if submit_button:
    # Encode the location using the same LabelEncoder as used during training
    le = joblib.load('label_encoder.joblib')
    location_encoded = le.transform([location])[0]

    # Create a numpy array from user input and location encoded value
    input_features = np.array([[beds, baths, area, location_encoded]])

    # Make a prediction using the saved decision tree model
    prediction = decision_tree_model.predict(input_features)

    # Display the prediction to the user
    st.write('The predicted price is: ${:.2f}'.format(prediction[0]))
