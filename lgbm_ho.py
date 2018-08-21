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
#	"learning_rate": 0.002,
#	"num_leaves": 60,
#	"feature_fraction": 0.5,
#	"bagging_fraction": 1,
#	"bagging_freq": 0,
	"min_data_in_leaf": 500,
	"min_sum_hessian_in_leaf": 1,
#	"lambda_l1": 0,
#	"lambda_l2": 0,
	"n_jobs": 2,
	"random_state": 456,
}

param_grid = {
	"learning_rate": [0.001*n for n in range(1, 11)],
	"feature_fraction": [0.3, 0.4, 0.5, 0.6, 0.7],
	"num_leaves": [32, 64, 128]
}

fit_params = {
	"early_stopping_rounds": 5
}

model_hyperopt(lgb.LGBMRegressor, 2017, init_params, param_grid, fit_params=fit_params, cv=3, max_evals=100, n_cv_jobs=3, verbose=1, out="results/lgbm_ho_1")

print("Done")
