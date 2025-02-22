import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the app
st.title("Data Science Model Training App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset
    st.write("Preview of the dataset:")
    st.write(df.head())
    
    # Select features and target
    st.sidebar.header("Select Features and Target")
    features = st.sidebar.multiselect("Select features", df.columns)
    target = st.sidebar.selectbox("Select target", df.columns)
    
    if features and target:
        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display the accuracy
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Display the model's coefficients (optional)
        st.write("Model Coefficients:")
        st.write(model.coef_)
        
    else:
        st.warning("Please select at least one feature and a target column.")
else:
    st.info("Please upload a CSV file to get started.")