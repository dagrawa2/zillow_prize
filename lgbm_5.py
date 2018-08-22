import numpy as np
import lightgbm as lgb
from models import model_run
from utils import *

np.random.seed(123)

class counter(object):

	def __init__(self):
		self.count = 1

	def __call__(self, dummy):
		print("[", self.count, "] . . . ")
		self.count += 1


init_params = {
	"boosting_type": "gbdt",
	"objective": "regression",
	"metric": "mae",
	"n_estimators": 500,
	"learning_rate": 0.002,
	"num_leaves": 64,
	"feature_fraction": 0.5,
	"bagging_fraction": 0.8,
	"bagging_freq": 10,
	"min_data_in_leaf": 500,
	"min_sum_hessian_in_leaf": 1,
	"lambda_l1": 0.001,
	"lambda_l2": 0.01,
	"importance_type": "gain",
	"n_jobs": -1,
	"random_state": 456,
}

fit_params = {
	"callbacks": [counter()]
}

model_run(lgb.LGBMRegressor, 2016, init_params, fit_params=fit_params, save_model=True, out="results/lgbm_5")
model_run(lgb.LGBMRegressor, 2017, init_params, fit_params=fit_params, save_model=True, out="results/lgbm_5")

print("Done")
