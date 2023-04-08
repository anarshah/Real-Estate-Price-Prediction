import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# load the trained model
model = joblib.load('decision_tree_model.joblib')

# load the dataset
df = pd.read_csv('zameen-property-data.csv')

# define the columns used to train the model
columns = ['bedrooms', 'bathrooms', 'area', 'location', 'city', 'purpose']

# get unique values for the "purpose" column
purpose_options = df['purpose'].unique()

# define a function to get user inputs and make predictions
def predict_price(bedrooms, bathrooms, area, location, city, purpose):
    # create a DataFrame with the user inputs
    data = pd.DataFrame([[bedrooms, bathrooms, area, location, city, purpose]], columns=columns)
    
    # encode non-numeric data
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
    
    # use the trained model to make a prediction
    predicted_price = model.predict(data)[0]
    
    # return the predicted price
    return predicted_price

# define the Streamlit app
def app():
    st.title('Zameen Property Price Predictor')
    
    # define input fields for the user to enter data
    bedrooms = st.number_input('Number of Bedrooms')
    bathrooms = st.number_input('Number of Bathrooms')
    area = st.number_input('Area (in square feet)')
    location = st.text_input('Location')
    city = st.text_input('City')
    
    # define a dropdown menu for the "purpose" input field
    purpose = st.selectbox('Purpose', purpose_options)
    
    # define a button to trigger the prediction
    if st.button('Predict Price'):
        # make a prediction using the user inputs
        predicted_price = predict_price(bedrooms, bathrooms, area, location, city, purpose)
        
        # display the predicted price to the user
        st.success(f'Predicted Price: {predicted_price:.2f} PKR')

# run the Streamlit app
if __name__ == '__main__':
    app()
