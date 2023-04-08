import streamlit as st
import joblib
import pandas as pd

# Load the label encoder and decision tree model
le = joblib.load('label_encoder.joblib')
model = joblib.load('decision_tree_model.joblib')

# Create a function to preprocess user input and make predictions
def predict_price(location, area, bedrooms, bathrooms):
    # Load the data and select the necessary columns
    data = pd.read_csv('zameen-property-data.csv')[['location', 'area', 'bedrooms', 'baths', 'price']]
    data['location'] = le.transform(data['location'])
    
    # Create a new DataFrame with the user input
    user_input = pd.DataFrame({
        'location': [location],
        'area': [area],
        'bedrooms': [bedrooms],
        'baths': [bathrooms]
    })
    user_input['location'] = le.transform(user_input['location'])
    
    # Merge the user input with the original data and select the necessary columns
    data = pd.concat([data, user_input])
    data = data[['location', 'area', 'bedrooms', 'baths']]
    
    # Separate the user input from the original data
    X = data.iloc[:-1]
    user_input = data.iloc[-1:]
    
    # Train a new label encoder on the 'location' column of the updated data
    le_new = LabelEncoder()
    le_new.fit(data['location'])
    
    # Encode the 'location' column of the user input using the new label encoder
    user_input['location'] = le_new.transform(user_input['location'])
    
    # Make a prediction using the model
    price = model.predict(user_input)[0]
    return price

# Create a Streamlit app
st.title("Property Price Predictor")

# Create input fields for user input
location = st.selectbox("Location", options=pd.read_csv('zameen-property-data.csv')['location'].unique())
area = st.number_input("Area (in Marla)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

# Create a button to make the prediction
if st.button("Predict Price"):
    price = predict_price(location, area, bedrooms, bathrooms)
    st.write("The predicted price is:", price)
