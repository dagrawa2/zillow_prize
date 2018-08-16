import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models import model_hyperopt
from utils import *

init_params = {
	"criterion": "mae",
	"min_samples_split": 500
}

param_grid = {
	"n_estimators": [10, 50], # , 100, 500],
	"max_features": [0.2, 0.4], # , 0.6, 0.8],
	"max_leaf_nodes": [8, 16, 32] # , 64]
}

model_hyperopt(RandomForestRegressor, 2016, init_params, param_grid, cv=3, verbose=6, out="results/rf")

print("Done")
