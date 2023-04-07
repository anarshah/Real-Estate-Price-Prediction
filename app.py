import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Load the data
df = pd.read_csv('zameen-property-data.csv')

# Remove unwanted columns
df.drop(['property_id', 'location_id', 'page_url', 'purpose', 'date_added', 'agency', 'agent'], axis=1, inplace=True)

# Convert area column to numeric
df['area'] = df['area'].str.replace('Marla', '').astype(float)

# Convert location column to categorical
df['location'] = pd.Categorical(df['location']).codes

# Convert city column to categorical
df['city'] = pd.Categorical(df['city']).codes

# Convert province_name column to categorical
df['province_name'] = pd.Categorical(df['province_name']).codes

# Convert property_type column to categorical
df['property_type'] = pd.Categorical(df['property_type']).codes

# Define a function to make a prediction
def predict_price(bedrooms, bathrooms, area):
    # Create a dictionary with the user input
    data = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area': area}
    
    # Create a DataFrame from the dictionary
    df_user_input = pd.DataFrame(data, index=[0])
    
    # Use the trained model to make a prediction on the user input
    predicted_price = model.predict(df_user_input)[0]
    
    return predicted_price

# Define the Streamlit app
def app():
    st.title("Property Price Predictor")
    
    # Get user input
    bedrooms = st.number_input("Enter number of bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Enter number of bathrooms", min_value=1, max_value=10, step=1)
    area = st.number_input("Enter area in square meters", min_value=1.0, max_value=1000.0, step=1.0)
    
    # Make a prediction
    predicted_price = predict_price(bedrooms, bathrooms, area)
    
    # Show the predicted price
    st.write(f"The predicted price is {predicted_price:.2f} PKR.")
    
# Run the app
if __name__ == '__main__':
    app()
