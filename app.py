import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


# Load the trained decision tree model
model = joblib.load('decision_tree_model.joblib')

# Load the label encoder for the location feature
le = joblib.load('label_encoder.joblib')

# Load the training data
train_data = pd.read_csv('zameen-property-data.csv')
X_train = train_data.drop('price', axis=1)
y_train = train_data['price']

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

    # Fit the model on the training data
    model.fit(X_train, y_train)

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
