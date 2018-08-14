import gc
import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
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


def model_hyperopt(model, year, init_params, param_grid, fit_params={}, cv=2, verbose=0, out=None):
	print("\n---\nCalling model_hyperopt on ", year, " data")
	time_0 = time.time()
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year)
	estimator = model(**init_params)
	model_grid = GridSearchCV(estimator, param_grid, cv=cv, verbose=verbose)
	print("Performing grid search")
	model_grid.fit(x_train, y_train, **fit_params)
	cv_results = {param: model_grid.cv_results_["param_"+param] for param in param_grid.keys()}
	cols = order_masked_params(cv_results, param_grid)
	cv_results.update({"mean_test_score": model_grid.cv_results_["mean_test_score"]})
	cols.append("mean_test_score")
	cv_results = pd.DataFrame.from_dict(cv_results)[cols]
	best = model_grid.best_params_
	best.update({"score": model_grid.best_score_})
	times = {"mean_fit_time": np.mean(model_grid.cv_results_["mean_fit_time"]), "total_time": time.time()-time_0}
	args = {"model": model.__name__, "year": year, "cv": cv}
	print("Saving results\n---\n")
	cv_results.to_csv(out+"/cv_results.csv", index=False)
	save_json(init_params, out+"/init_params.json")
	save_json(param_grid, out+"/param_grid.json")
	save_json(fit_params, out+"/fit_params.json")
	save_json(args, out+"/args.json")
	save_json(best, out+"/best.json")
	save_json(times, out+"/times.json")


def model_pred(model, year, init_params, fit_params={}, save_model=False, out=None):
	print("\n---\nCalling model_pred on year ", year, " data")
	time_0 = time.time()
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year)
	estimator = model(**init_params)
	print("Training")
	estimator.fit(x_train, y_train, **fit_params)
	del x_train; gc.collect()
	del y_train; gc.collect()
	if save_model:
		print("Saving model")
		joblib.dump(estimator, out+"/model.jl")
	print("Loading test data")
	x_test = load_test(year)
	train_columns = load_pickle("preprocessed/train_"+str(year)+"_cols.list")
	month_col_index = train_columns.index("month")
	print("Loading submission")
	if year == 2016:
		sub = pd.read_csv("data/sample_submission.csv")
	else:
		sub = pd.read_csv(out+"/preds.csv")
	time_1 = time.time()
	print("Predicting")
	for month in [10, 11, 12]:
		print("\tmonth = ", month)
		x_test[:,month_col_index] = month
		sub[str(year)+str(month)] = estimator.predict(x_test)
	time_2 = time.time()
	print("Saving results\n---\n")
	sub.to_csv(out+"/preds.csv", index=False, float_format='%.4f')
	times = {"pred_time": time_2-time_1, "total_time": time.time()-time_0}
	args = {"model": model.__name__, "year": year}
	save_json(init_params, out+"/init_params.json")
	save_json(fit_params, out+"/fit_params.json")
	save_json(args, out+"/args.json")
	save_json(times, out+"/times.json")


def lgbm_pred(year, init_params, fit_params={}, save_model=False, out=None):
	print("\n---\nCalling model_pred on year ", year, " data")
	time_0 = time.time()
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year)
	print("Training")
	estimator = lgb.train(init_params, lgb.Dataset(x_train, label=y_train), **fit_params)
	del x_train; gc.collect()
	del y_train; gc.collect()
	if save_model:
		print("Saving model")
		save_json(estimator.dump_model(), out+"/model.json")
	print("Loading test data")
	x_test = load_test(year)
	train_columns = load_pickle("preprocessed/train_"+str(year)+"_cols.list")
	month_col_index = train_columns.index("month")
	print("Loading submission")
	if year == 2016:
		sub = pd.read_csv("data/sample_submission.csv")
	else:
		sub = pd.read_csv(out+"/preds.csv")
	time_1 = time.time()
	print("Predicting")
	for month in [10, 11, 12]:
		print("\tmonth = ", month)
		x_test[:,month_col_index] = month
		sub[str(year)+str(month)] = estimator.predict(x_test)
	time_2 = time.time()
	print("Saving results\n---\n")
	sub.to_csv(out+"/preds.csv", index=False, float_format='%.4f')
	times = {"pred_time": time_2-time_1, "total_time": time.time()-time_0}
	args = {"year": year}
	save_json(init_params, out+"/init_params.json")
	save_json(fit_params, out+"/fit_params.json")
	save_json(args, out+"/args.json")
	save_json(times, out+"/times.json")

