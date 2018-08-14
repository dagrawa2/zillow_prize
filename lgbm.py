import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models import lgbm_pred
from utils import *

init_params = {
	"boosting_type": "gbdt",
	"objective": "regression",
	"metric": "mae",
	"learning_rate": 0.002,
	"sub_feature": 0.5,
	"num_leaves": 60,
	"min_data": 500,
	"min_hessian": 1
}

fit_params = {
	"num_boost_round": 200
}

lgbm_pred(2016, init_params, fit_params=fit_params, save_model=True, out="results/lgbm")

print("Done")
