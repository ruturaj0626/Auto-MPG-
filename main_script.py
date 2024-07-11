import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing import preprocess_data
from modeling import train_model
from metrics import evaluate_model, print_evaluation
from visualization import plot_feature_importances, plot_predictions
import numpy as np

# Load data
df = pd.read_csv(r'D:\Antern\Make_Pro_Prod\AutoMPG\auto-mpg.csv')

# Preprocess data
df = preprocess_data(df)

# Define features and target
X = df.drop(['mpg'], axis=1)
y = df['mpg']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
mse, r2 = evaluate_model(model, X_test, y_test)
print_evaluation(mse, r2)

# Visualize results
plot_feature_importances(model, X.columns)
plot_predictions(y_test, model.predict(X_test))
