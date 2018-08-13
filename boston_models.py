import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import *
	
def order_masked_params(D, param_grid):
	inds_params = {}
	for param in D.keys():
		i = np.argwhere(np.ma.getdata(D[param])==param_grid[param][1])[0,0]
		inds_params.update({i: param})
	inds = list(inds_params.keys())
	inds.sort()
	inds.reverse()
	return [inds_params[i] for i in inds]


def model_hyperopt(model, x_train, y_train, fixed_params, param_grid, cv=2, verbose=0, out=None, **kwargs):
	time_0 = time.time()
#	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	estimator = model(**fixed_params)
	model_grid = GridSearchCV(estimator, param_grid, cv=cv, verbose=verbose)
	model_grid.fit(x_train, y_train, **kwargs)
	cv_results = {param: model_grid.cv_results_["param_"+param] for param in param_grid.keys()}
	cols = order_masked_params(cv_results, param_grid)
	cv_results.update({"mean_test_score": model_grid.cv_results_["mean_test_score"]})
	cols.append("mean_test_score")
	cv_results = pd.DataFrame.from_dict(cv_results)[cols]
	best = model_grid.best_params_
	best.update({"score": model_grid.best_score_})
	times = {"mean_fit_time": np.mean(model_grid.cv_results_["mean_fit_time"]), "total_time": time.time()-time_0}
	kwargs.update({"cv": cv})
	cv_results.to_csv(out+"/cv_results.csv", index=False)
	save_json(param_grid, out+"/param_grid.json")
	save_json(kwargs, out+"/other_kwargs.json")
	save_json(best, out+"/best.json")
	save_json(times, out+"/times.json")


def model_pred(model, x_train, y_train, params, out=None):
	time_0 = time.time()
#	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	estimator = model(**params)
	estimator.fit(x_train, y_train)
	y_pred = estimator.predict(x_test)

	cv_results = {param: model_grid.cv_results_["param_"+param] for param in param_grid.keys()}
	cols = order_masked_params(cv_results, param_grid)
	cv_results.update({"mean_test_score": model_grid.cv_results_["mean_test_score"]})
	cols.append("mean_test_score")
	cv_results = pd.DataFrame.from_dict(cv_results)[cols]
	best = model_grid.best_params_
	best.update({"score": model_grid.best_score_})
	times = {"mean_fit_time": np.mean(model_grid.cv_results_["mean_fit_time"]), "total_time": time.time()-time_0}
	kwargs.update({"cv": cv})
	cv_results.to_csv(out+"/cv_results.csv", index=False)
	save_json(param_grid, out+"/param_grid.json")
	save_json(kwargs, out+"/other_kwargs.json")
	save_json(best, out+"/best.json")
	save_json(times, out+"/times.json")


