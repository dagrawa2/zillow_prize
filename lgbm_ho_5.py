import numpy as np
import lightgbm as lgb
from models import model_hyperopt
from utils import *

np.random.seed(123)

init_params = {
	"boosting_type": "gbdt",
	"objective": "regression",
	"metric": "mae",
	"n_estimators": 500,
	"learning_rate": 0.002,
	"num_leaves": 64,
	"feature_fraction": 0.5,
#	"bagging_fraction": 1,
	"bagging_freq": 10,
	"min_data_in_leaf": 500,
	"min_sum_hessian_in_leaf": 1,
#	"lambda_l1": 0,
#	"lambda_l2": 0,
	"n_jobs": 3,
	"random_state": 456,
}

param_grid = {
	"bagging_fraction": [0.2, 0.4, 0.6, 0.8, 1],
	"lambda_l1": [0, 0.001, 0.01],
	"lambda_l2": [0, 0.001, 0.01]
}

fit_params = {
	"early_stopping_rounds": 5
}

model_hyperopt(lgb.LGBMRegressor, 2016, init_params, param_grid, fit_params=fit_params, cv=3, max_evals=45, n_cv_jobs=3, verbose=1, out="results/lgbm_ho_6")

print("Done")
