from sklearn.tree import DecisionTreeRegressor
import joblib

# load the saved model and label encoder
model = joblib.load('decision_tree_model.joblib')
encoder = joblib.load('label_encoder.joblib')

# define a function to predict the price
def predict_price(location, sqft, bedrooms, bathrooms):
    # encode the location using the label encoder
    location_encoded = encoder.transform([location])[0]

    # create a data frame with the input features
    input_data = [[location_encoded, sqft, bedrooms, bathrooms]]
    input_df = pd.DataFrame(input_data, columns=['location_encoded', 'sqft', 'bedrooms', 'bathrooms'])

    # predict the price using the fitted model
    prediction = model.predict(input_df)

    return prediction
