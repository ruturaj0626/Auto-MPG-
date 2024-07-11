from sklearn.ensemble import RandomForestRegressor
import numpy as np
from hyperparameters import RANDOM_FOREST_PARAMS

def train_model(X, y):
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    return score
