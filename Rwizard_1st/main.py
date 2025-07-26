import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def data_load(csv_dir):
    # Load dataset
    df = pd.read_csv(csv_dir)
    return df

def data_preprocess(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Convert categorical variables to numerical
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def train_model(df, test_size):
    # Split the dataset into features and target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2


dataset_path = '/Rwizard_1st/datasets/dataset_antoine.csv' 

if not os.path.exists(os.getcwd() + dataset_path):
    raise FileNotFoundError("Dataset file not found at the specified path.")

df=data_load(os.getcwd() + dataset_path)

print ("Dataset loaded successfully.")
print (df.head())

