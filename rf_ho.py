import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from models import model_hyperopt

init_params = {
	"criterion": "mae",
#	"n_estimators": 10,
	"min_samples_split": 500
}

param_grid = {
	"n_estimators": [50, 100, 150, 200],
	"max_features": [0.3, 0.5, 0.7],
	"max_leaf_nodes": [16, 32, 64]
}

fit_params = {}

model_hyperopt(RandomForestRegressor, 2016, init_params, param_grid, fit_params=fit_params, cv=3, max_evals=10, n_cv_jobs=3, verbose=1, out="results/rf_ho_2016")

print("Finished")
