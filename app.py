import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# load the saved model and label encoder
# Load the dataset
df = pd.read_csv('zameen-property-data.csv')

# encode non-numerical data
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

model = joblib.load('decision_tree_model.joblib')

def predict_price(model, city, area_sqft, bedrooms, baths):
    # Check if the city is in the LabelEncoder's classes
    if city in le.classes_:
        # Encode the city using the same LabelEncoder used during training
        city_encoded = le.transform([city])[0]
    else:
        # Set the city_encoded to the mode of the 'city' column if the city is not in the training dataset
        city_encoded = df['city'].mode().iloc[0]

    # Create a new DataFrame with the same columns as your training data
    input_data = pd.DataFrame(columns=X_train.columns)
    
    # Fill in the user's input data
    input_data.loc[0, 'city'] = city_encoded
    input_data.loc[0, 'area'] = area_sqft
    input_data.loc[0, 'bedrooms'] = bedrooms
    input_data.loc[0, 'baths'] = baths

    # Set other columns to their mean or mode values, as appropriate
    for col in input_data.columns:
        if col not in ['city', 'area', 'bedrooms', 'baths']:
            if input_data[col].dtype == 'object':
                input_data[col] = df[col].mode().iloc[0]
            else:
                input_data[col] = df[col].mean()

    # Make a prediction using the trained model
    price_prediction = model.predict(input_data)
    
    return price_prediction[0]


# Set up the Streamlit app
st.title("Property Price Predictor")

# Create input fields for user input
city = st.text_input("City")
area_sqft = st.number_input("Area (in square feet)", min_value=0, value=900, step=1)
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=2, step=1)
baths = st.number_input("Number of bathrooms", min_value=0, value=2, step=1)

# Create a button to trigger the price prediction
if st.button("Predict Price"):
    # Call the predict_price function with the user input
    predicted_price = predict_price(model, city, area_sqft, bedrooms, baths)
    
    # Display the predicted price
    st.write(f"Predicted property price: {predicted_price:.2f} PKR.")
