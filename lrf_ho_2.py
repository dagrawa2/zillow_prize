import numpy as np
import lightgbm as lgb
from models import model_hyperopt
from utils import *

np.random.seed(123)

class counter(object):

	def __init__(self):
		self.count = 1

	def __call__(self, dummy):
		print("[", self.count, "] . . . ")
		self.count += 1


init_params = {
	"boosting_type": "rf",
	"objective": "regression",
	"metric": "mae",
#	"n_estimators": 100,
	"learning_rate": 0.002,
#	"num_leaves": 128,
#	"feature_fraction": 0.5,
#	"bagging_fraction": 0.4,
	"bagging_freq": 1,
	"min_data_in_leaf": 500,
	"min_sum_hessian_in_leaf": 1,
#	"lambda_l1": 0,
#	"lambda_l2": 0,
	"importance_type": "gain",
	"n_jobs": -1,
	"random_state": 456,
}

param_grid = {
	"n_estimators": [50, 100, 150, 200],
	"num_leaves": [32, 64, 128],
	"feature_fraction": [0.8, 0.16, 0.32, 0.64],
	"bagging_fraction": [0.2, 0.3, 0.4, 0.5]
}

fit_params = {
#	"callbacks": [counter()]
}

model_hyperopt(lgb.LGBMRegressor, 2016, init_params, param_grid, fit_params=fit_params, cv=3, max_evals=100, n_cv_jobs=3, verbose=1, out="results/lrf_ho_2")

print("Done")
