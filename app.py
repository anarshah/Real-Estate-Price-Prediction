import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# Load the trained decision tree model
model = joblib.load('decision_tree_model.joblib')

# Load the label encoder for the location feature
le = joblib.load('label_encoder.joblib')

# Load the training data
df = pd.read_csv('zameen-property-data.csv')

# Split the data into features and target variables
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to preprocess the input data and make predictions
def predict_price(location, sqft, bedrooms, bathrooms):
    # Encode the location feature
    location_encoded = le.transform([location])[0]

    # Create a dictionary with the input data
    input_dict = {
        'location': location_encoded,
        'total_sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms
    }

    # Create a pandas DataFrame from the input dictionary
    input_df = pd.DataFrame([input_dict])

    # Make predictions on the input data
    prediction = model.predict(input_df)

    return prediction


# Create the Streamlit app
st.title('Real Estate Price Prediction')

# Add input fields for the user to enter the property details
location = st.selectbox('Location', ['Gulberg', 'DHA', 'Bahria Town', 'Safari Villas'])
sqft = st.number_input('Total sqft')
bedrooms = st.number_input('Bedrooms')
bathrooms = st.number_input('Bathrooms')

# Add a button to make predictions on the input data
if st.button('Predict'):
    # Call the predict_price function with the input data
    prediction = predict_price(location, sqft, bedrooms, bathrooms)
    st.success(f'The predicted price is {prediction:.2f} PKR.')
