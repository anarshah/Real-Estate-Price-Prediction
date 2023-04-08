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
def predict_price(data):
    # load the trained model
    model = joblib.load('decision_tree_model.joblib')
    
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
    
    # load the dataset
    df = pd.read_csv('zameen-property-data.csv')
    
    # define input fields for the user to enter data
    city = st.selectbox('City', df['city'].unique())
    
    # get unique values for the "location" column based on the selected city
    location_options = df[df['city'] == city]['location'].unique()
    
    location = st.selectbox('Location', location_options)
    
    area = st.number_input('Area (in square feet)')
    bedrooms = st.number_input('Number of Bedrooms')
    bathrooms = st.number_input('Number of Bathrooms')
    
    purpose = st.selectbox('Purpose', df['purpose'].unique())
    property_type = st.selectbox('Property Type', df['property_type'].unique())
    
    # define input fields for the remaining columns to use as filters
    price = st.slider('Price (in PKR)', min_value=df['price'].min(), max_value=df['price'].max(), step=100000, value=float(df['price'].mean()))
    baths = st.slider('Number of Baths', min_value=df['baths'].min(), max_value=df['baths'].max(), step=1, value=df['baths'].mean())
    date_added = st.date_input('Date Added', df['date_added'].min(), df['date_added'].max(), df['date_added'].mean())
    
    # define a button to trigger the prediction
    if st.button('Predict Price'):
        # create a DataFrame with the user inputs
        data = pd.DataFrame({
            'city': [city],
            'location': [location],
            'area': [area],
            'bedrooms': [bedrooms],
            'baths': [bathrooms],
            'purpose': [purpose],
            'property_type': [property_type],
            'price': [price],
            'date_added': [date_added],
            'property_id': [0],
            'location_id': [0],
            'page_url': [''],
            'province_name': [''],
            'latitude': [0.0],
            'longitude': [0.0],
            'agency': [''],
            'agent': ['']
        })
        
        # make a prediction using the user inputs
        predicted_price = predict_price(data)
        
        # display the predicted price to the user
        st.success(f'Predicted Price: {predicted_price:.2f} PKR')


# run the Streamlit app
if __name__ == '__main__':
    app()
