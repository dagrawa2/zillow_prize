import numpy as np
import lightgbm as lgb
from models import model_hyperopt
from utils import *

init_params = {
	"boosting_type": "gbdt",
	"objective": "regression",
	"metric": "mae",
	"n_estimators": 500,
#	"learning_rate": 0.002,
#	"sub_feature": 0.5,
#	"num_leaves": 60,
	"min_data": 500,
	"min_hessian": 1,
	"n_jobs": 2
}

param_grid = {
	"learning_rate": [0.001*n for n in range(1, 11)],
	"sub_feature": [0.3, 0.4, 0.5, 0.6, 0.7],
	"num_leaves": [32, 64, 128]
}

fit_params = {
	"early_stopping_rounds": 5
}

model_hyperopt(lgb.LGBMRegressor, 2016, init_params, param_grid, fit_params=fit_params, cv=3, max_evals=100, n_cv_jobs=3, verbose=1, out="results/lgbm_ho2")

print("Done")
