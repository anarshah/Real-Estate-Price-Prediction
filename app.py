import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    data = pd.read_csv("zameen-property-data.csv")
    data.drop(["property_id", "location_id", "page_url", "latitude", "longitude", "date_added", "agency", "agent"], axis=1, inplace=True)
    return data

def preprocess_data(data):
    # encode categorical features
    encoder = LabelEncoder()
    data["property_type"] = encoder.fit_transform(data["property_type"])
    data["location"] = encoder.fit_transform(data["location"])
    data["city"] = encoder.fit_transform(data["city"])
    data["province_name"] = encoder.fit_transform(data["province_name"])
    data["purpose"] = encoder.fit_transform(data["purpose"])
    
    # split data into X and y
    X = data.drop(["price"], axis=1)
    y = data["price"]
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # make predictions on test data
    y_pred = model.predict(X_test)
    
    # calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    return mae, rmse, r2

def main():
    st.set_page_config(page_title="Real Estate Price Prediction", page_icon=":house:", layout="wide")
    st.title("Real Estate Price Prediction")
    
    # load data
    data = load_data()
    
    # preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # train model
    model = train_model(X_train, y_train)
    
    # evaluate model
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # display results
    st.write("Mean Absolute Error:", mae)
    st.write("Root Mean Squared Error:", rmse)
    st.write("R2 Score:", r2)

if __name__ == "__main__":
    main()
