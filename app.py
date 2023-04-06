import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("property_dataset.csv")
    return data

# Define the training and evaluation process
def train_and_evaluate_model(data):
    # Prepare the data
    X = data.drop(["price"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, rmse, r2

# Define the main function
def main():
    # Load the data
    data = load_data()
    
    # Show the dataset on Streamlit
    st.write("## Property Dataset")
    st.write(data)
    
    # Train and evaluate the model
    st.write("## Model Training and Evaluation")
    model, mae, rmse, r2 = train_and_evaluate_model(data)
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    
    # Show the feature importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": model.feature_importances_,
        "Importance": data.drop(["price"], axis=1).columns
    }).sort_values("Feature", ascending=False)
    st.bar_chart(feature_importance.head(10))
    
if __name__ == "__main__":
    main()
