import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("zameen_property_model.joblib")

# Define the preprocess_area function
def preprocess_area(area_str):
    
    def marla_to_sqft(marla):
        return marla * 225

    def kanal_to_sqft(kanal):
        return kanal * 5062.5

    area_value, area_unit = area_str.split()
    area_value = float(area_value.replace(",", ""))

    if area_unit.lower() == "marla":
        area_value = marla_to_sqft(area_value)
    elif area_unit.lower() == "kanal":
        area_value = kanal_to_sqft(area_value)
    # Add more units if needed

    return area_value

# Load the data for province, city, and location options
data = pd.read_csv("zameen-property-data.csv")
provinces = data["province_name"].unique()

# Define the options for province, city, and location
province_option = st.selectbox("Select Province", provinces)
city_option = st.selectbox("Select City", data[data["province_name"] == province_option]["city"].unique())
location_option = st.selectbox("Select Location", data[data["city"] == city_option]["location"].unique())

# Define the input options for area, bedrooms, baths, purpose, and property type
area_input = st.number_input("Area in Square Feet")
bedrooms_input = st.number_input("Number of Bedrooms")
baths_input = st.number_input("Number of Baths")
purpose_input = st.selectbox("Select Purpose", ["Buy", "Rent"])
property_type_input = st.selectbox("Select Property Type", ["House", "Flat"])

# Define the input dataframe based on the user inputs
input_df = pd.DataFrame({
    "province_name": [province_option],
    "city": [city_option],
    "location": [location_option],
    "area": [preprocess_area(str(area_input) + " Square Feet")],
    "bedrooms": [bedrooms_input],
    "baths": [baths_input],
    "purpose": [purpose_input],
    "property_type": [property_type_input]
})

# Encode categorical features
encoder = joblib.load("encoder.joblib")
input_df['property_type'] = encoder.transform(input_df['property_type'])
input_df['location'] = encoder.transform(input_df['location'])
input_df['city'] = encoder.transform(input_df['city'])
input_df['province_name'] = encoder.transform(input_df['province_name'])
input_df['purpose'] = encoder.transform(input_df['purpose'])

# Make predictions
prediction = model.predict(input_df)

# Display the predicted price to the user
st.write("The predicted price is: ", prediction[0])
