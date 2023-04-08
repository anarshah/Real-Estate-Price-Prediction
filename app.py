import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# load the trained model
model = joblib.load('decision_tree_model.joblib')

# load the dataset
df = pd.read_csv('zameen-property-data.csv')

# define the columns used to train the model
columns = ['bedrooms', 'bathrooms', 'area', 'location', 'city', 'purpose', 'property_type']

# get unique values for the "city" and "purpose" columns
city_options = df['city'].unique()
purpose_options = df['purpose'].unique()

# define a function to get the unique values for the "location" column based on the selected city
def get_location_options(city):
    location_options = df[df['city'] == city]['location'].unique()
    return location_options

# define a function to get user inputs and make predictions
def predict_price(bedrooms, bathrooms, area, location, city, purpose, property_type):
    # create a DataFrame with the user inputs
    data = pd.DataFrame({
        'city': [city],
        'location': [location],
        'area': [area],
        'bedrooms': [bedrooms],
        'baths': [bathrooms],
        'purpose': [purpose],
        'property_type': [property_type]
    })
    
    # encode non-numerical data
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
    
    # make a prediction using the user inputs
    predicted_price = model.predict(data)[0]
    
    return predicted_price

# define the Streamlit app
def app():
    st.title('Zameen Property Price Predictor')
    
    # define input fields for the user to enter data
    city = st.selectbox('City', city_options)
    
    # get unique values for the "location" column based on the selected city
    location_options = get_location_options(city)
    
    location = st.selectbox('Location', location_options)
    
    area = st.number_input('Area (in square feet)')
    bedrooms = st.number_input('Number of Bedrooms')
    bathrooms = st.number_input('Number of Bathrooms')
    
    purpose = st.selectbox('Purpose', purpose_options)
    property_type = st.selectbox('Property Type', df['property_type'].unique())
    
    # define a button to trigger the prediction
    if st.button('Predict Price'):
        # make a prediction using the user inputs
        predicted_price = predict_price(bedrooms, bathrooms, area, location, city, purpose, property_type)
        
        # display the predicted price to the user
        st.success(f'Predicted Price: {predicted_price:.2f} PKR')

# run the Streamlit app
if __name__ == '__main__':
    app()
