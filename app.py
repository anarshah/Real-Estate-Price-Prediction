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
    # Load the saved model
    model = load_model()

    # Load the dataset
    df = load_data()

    # Remove any rows with missing data
    df.dropna(inplace=True)

    # Exclude non-numeric columns and columns not needed for prediction
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('property_id') # exclude ID column
    numeric_cols.remove('location_id') # exclude ID column
    numeric_cols.remove('latitude') # exclude location column
    numeric_cols.remove('longitude') # exclude location column
    numeric_cols.remove('date_added') # exclude date column
    data = df[numeric_cols]

    # Calculate mean values for imputation
    mean_values = data.mean()

    # Fill missing values with mean values
    df.fillna(mean_values, inplace=True)

    # Select subset of data based on input values
    subset = df[(df['city'] == city) & (df['location'] == location) & (df['area'] == area) & (df['bedrooms'] == bedrooms) & (df['baths'] == bathrooms) & (df['purpose'] == purpose) & (df['property_type'] == property_type)]

    # Make prediction
    predicted_price = model.predict(subset[numeric_cols])[0]

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
    price = st.slider('Price (in PKR)', 
                  min_value=int(df['price'].min()), 
                  max_value=int(df['price'].max()), 
                  step=100000, 
                  value=int(df['price'].mean()))

    baths = st.slider('Number of Baths', min_value=int(df['baths'].min()), max_value=int(df['baths'].max()), step=1, value=int(df['baths'].mean()))
    # drop rows with non-numeric values in "date_added" column
    df = df[pd.to_numeric(df['date_added'], errors='coerce').notnull()]

    # create date input widget
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
