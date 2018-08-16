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
	"min_data": 10,
	"min_hessian": 1,
	"n_jobs": 2
}

param_grid = {
	"learning_rate": [0.002, 0.02, 0.2],
	"num_leaves": [4, 8, 16]
}

fit_params = {
	"early_stopping_rounds": 2
}

model_hyperopt(lgb.LGBMRegressor, 0, init_params, param_grid, fit_params=fit_params, cv=2, max_evals=10, n_cv_jobs=2, verbose=1, out="results/boston")

print("Done")
