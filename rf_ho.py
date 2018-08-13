import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models import model_hyperopt
from utils import *

print("Loading data")
x_train, y_train = load_train(2016)

fixed_params = {
	"criterion": "mae",
	"min_samples_split": 500
}

param_grid = {
	"n_estimators": [10, 50, 100, 500],
	"max_features": [0.2, 0.4, 0.6, 0.8],
	"max_leaf_nodes": [8, 16, 32, 64]
}

print("Performing HO")
model_hyperopt(RandomForestRegressor, x_train, y_train, fixed_params, param_grid, cv=5, verbose=1, out="results/rf")

print("Done")
