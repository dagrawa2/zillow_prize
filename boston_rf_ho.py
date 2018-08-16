import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from models import model_hyperopt

init_params = {
	"criterion": "mae",
	"n_estimators": 10,
	"min_samples_split": 50
}

param_grid = {
	"max_features": [0.25, 0.5, 0.75],
	"max_leaf_nodes": [8, 16, 32]
}

model_hyperopt(RandomForestRegressor, 0, init_params, param_grid, fit_params={}, cv=2, max_evals=10, verbose=1, out="results/rf_ho")

print("Finished")
