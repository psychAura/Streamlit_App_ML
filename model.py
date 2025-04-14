import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

# Load the saved model
model = joblib.load('model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')  # If you saved the scaler too
vec = joblib.load('vec.pkl')

# Function for preprocessing input data
def preprocess_input(input_data):
    # Preprocess numerical features
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                        "PaperlessBilling", "PaymentMethod", "Dependents", "Partner"]

    # Standardize numerical columns
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

   # Transform categorical columns using DictVectorizer
    input_data_dict = input_data[categorical_cols].to_dict(orient="records")
    input_data_categorical = vec.transform(input_data_dict)

    # Convert the transformed categorical columns back to a DataFrame and concatenate with original data
    input_data_categorical_df = pd.DataFrame(input_data_categorical, columns=vec.get_feature_names_out())
    
    # Remove the original categorical columns and add the transformed ones
    input_data = input_data.drop(columns=categorical_cols)
    input_data = pd.concat([input_data, input_data_categorical_df], axis=1)

    return input_data

# Streamlit UI to get user input and make predictions
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn')

# Collecting user input
tenure = st.number_input('Tenure', min_value=0, max_value=100, value=1)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=1500.0)
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])

phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "SeniorCitizen": [1 if senior_citizen == 'Yes' else 0],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "Dependents": [dependents],
    "Partner": [partner]
})
# Button to trigger prediction
if st.button('Predict Churn'):
    # Preprocess the input data
    preprocessed_input = preprocess_input(input_data)

    # Predict the churn outcome
    prediction = model.predict(preprocessed_input)

    # Show the result
    if prediction == 1:
        st.success("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")
