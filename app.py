import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

# load the saved model and label encoder
model = joblib.load('decision_tree_model.joblib')
encoder = joblib.load('label_encoder.joblib')

# define a function to predict the price
def predict_price(location, sqft, bedrooms, bathrooms):
    # encode the location using the label encoder
    location_encoded = encoder.transform([location])[0]

    # create a data frame with the input features
    input_data = [[location_encoded, sqft, bedrooms, bathrooms]]
    input_df = pd.DataFrame(input_data, columns=['location_encoded', 'sqft', 'bedrooms', 'bathrooms'])
    
    # Debugging code
    print(f"input_df shape: {input_df.shape}")
    print(f"model input shape: {model.tree_.n_features}")

    # predict the price using the fitted model
    prediction = model.predict(input_df)

    return prediction

# set up the Streamlit app
st.set_page_config(page_title='Real Estate Price Prediction', page_icon=':money_with_wings:')
st.title('Real Estate Price Prediction')

# define the input fields
location = st.selectbox('Location', ['Gulberg', 'DHA', 'Bahria Town', 'Model Town', 'Johar Town'])
sqft = st.number_input('Square Feet')
bedrooms = st.number_input('Bedrooms')
bathrooms = st.number_input('Bathrooms')

# make the prediction when the 'Predict' button is clicked
if st.button('Predict'):
    prediction = predict_price(location, sqft, bedrooms, bathrooms)
    st.success(f'The predicted price is {prediction[0]:,.0f} PKR.')
