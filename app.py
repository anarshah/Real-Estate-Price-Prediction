import streamlit as st
import joblib
import pandas as pd

# load the trained model
model = joblib.load('decision_tree_model.joblib')

# create the UI elements to get user input
bedrooms = st.sidebar.slider('Number of bedrooms', 1, 10, 1)
bathrooms = st.sidebar.slider('Number of bathrooms', 1, 10, 1)
area = st.sidebar.slider('Area in square feet', 500, 10000, 1000)
location = st.sidebar.selectbox('Location', ['Gulberg', 'DHA', 'Johar Town', 'Bahria Town'])

# create a dictionary of the user input
input_dict = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area': area, 'location': location}

# convert the dictionary to a dataframe
input_df = pd.DataFrame([input_dict])

# make predictions using the loaded model
prediction = model.predict(input_df)

# display the predicted price to the user
st.write('The predicted price is:', prediction[0])
