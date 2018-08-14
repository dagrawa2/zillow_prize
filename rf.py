import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models import model_pred
from utils import *

init_params = {
	"criterion": "mae",
	"n_estimators": 10,
	"min_samples_split": 500,
	"max_features": 0.4,
	"max_leaf_nodes": 8
}

model_pred(RandomForestRegressor, 2016, init_params, save_model=True, out="results/rf")

print("Done")
