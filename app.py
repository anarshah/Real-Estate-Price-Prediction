import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# load the trained model
model = joblib.load('decision_tree_model.joblib')

# load the dataset
df = pd.read_csv('zameen-property-data.csv')

# drop unnecessary columns
df = df.drop(['property_id', 'page_url', 'city', 'province_name', 'date_added', 'agency', 'agent'], axis=1)

# encode non-numerical data
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# impute missing values
imputer = SimpleImputer()
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# define the columns used to train the model
columns = ['bedrooms', 'bathrooms', 'area', 'location', 'purpose', 'property_type']

# get unique values for the "location" column based on the selected city
def get_location_options(city):
    location_options = df[df['city'] == city]['location'].unique()
    return location_options

# define a function to get user inputs and make predictions
def predict_price(bedrooms, bathrooms, area, location, city, purpose, property_type):
    
    # create a DataFrame with the input data
    data = pd.DataFrame({'beds': [bedrooms], 'baths': [bathrooms], 'area': [area], 'location_id': [location], 'city_id': [city], 'purpose_id': [purpose], 'property_type_id': [property_type]})
    
    # rename the 'baths' column to 'bathrooms'
    data = data.rename(columns={'baths': 'bathrooms'})
    
    # impute missing values using the trained imputer
    data = pd.DataFrame(imputer.transform(data), columns=data.columns)
    
    # make the price prediction using the trained model
    predicted_price = model.predict(data)[0]
    
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
    
    # define a button to trigger the prediction
    if st.button('Predict Price'):
        # make a prediction using the user inputs
        predicted_price = predict_price(bedrooms, bathrooms, area, location, city, purpose, property_type)
        
        # display the predicted price to the user
        st.success(f'Predicted Price: {predicted_price:.2f} PKR')


# run the Streamlit app
if __name__ == '__main__':
    app()
