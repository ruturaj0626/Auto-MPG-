import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df):
    # Data cleaning and preprocessing steps
    df['horsepower'] = df['horsepower'].replace('?', np.nan).astype(float)
    
    # Drop 'car name' column
    df = df.drop(['car name'], axis=1)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df['horsepower'] = imputer.fit_transform(df[['horsepower']])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df
