import numpy as np
import lightgbm as lgb
from models import model_hyperopt
from utils import *

init_params = {
	"boosting_type": "gbdt",
	"objective": "regression",
	"metric": "mae",
	"n_estimators": 10,
##	"learning_rate": 0.002,
	"sub_feature": 0.5,
#	"num_leaves": 60,
	"min_data": 500,
	"min_hessian": 1
}

param_grid = {
	"learning_rate": [0.002, 0.02],
	"num_leaves": [30, 60]
}

fit_params = {
#	"early_stopping_rounds": 5
}

model_hyperopt(lgb.LGBMRegressor, 2016, init_params, param_grid, fit_params=fit_params, cv=3, verbose=6, out="results/lgbm_ho")

print("Done")
