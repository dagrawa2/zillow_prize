import numpy as np
import lightgbm as lgb
from models import model_run
from utils import *

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
	"n_estimators": 400,
	"learning_rate": 0.002,
	"sub_feature": 0.5,
	"num_leaves": 60,
	"min_data": 500,
	"min_hessian": 1,
	"importance_type": "gain",
	"n_jobs": -1
}

fit_params = {
	"callbacks": [counter()]
}

model_run(lgb.LGBMRegressor, 2017, init_params, fit_params=fit_params, save_model=True, out="results/lgbm")

print("Done")
