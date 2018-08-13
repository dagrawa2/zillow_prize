import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from boston_models import model_hyperopt

boston = load_boston()
x_train = boston.data
y_train = boston.target

fixed_params = {
	"criterion": "mae",
	"n_estimators": 10,
	"min_samples_split": 50
}

param_grid = {
	"max_features": [0.25, 0.5, 0.75],
	"max_leaf_nodes": [8, 16, 32]
}

model_hyperopt(RandomForestRegressor, x_train, y_train, fixed_params, param_grid, cv=5, verbose=1, out="rf")
print("Finished")
