import gc
import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
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


def model_gridsearch(model, year, init_params, param_grid, fit_params={}, cv=2, verbose=0, out=None):
	print("\n---\nCalling model_gridsearch on ", year, " data")
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

def model_hyperopt(model, year, init_params, param_grid, fit_params={}, cv=2, max_evals=10, n_cv_jobs=1, verbose=0, pca=False, out=None):
	if year == 0:
		print("\n---\nCalling model_hyperopt on Boston data")
	else:
		print("\n---\nCalling model_hyperopt on ", year, " data")
	time_0 = time.time()
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	out = out + "/" + str(year)
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year, pca=pca)
	param_list = list(param_grid.keys())
	param_list.sort()
	def objective(params):
		if verbose > 0: print("[ ]  (", "  ".join([param+"="+str(params[param]) for param in params]), ")")
		time_start = time.time()
		estimator = model(**init_params, **params)
		loss = cross_val_score(estimator, x_train, y_train, scoring=make_scorer(mean_absolute_error, greater_is_better=True), cv=cv, fit_params=fit_params, n_jobs=n_cv_jobs)
		if "early_stopping_rounds" in fit_params:
			loss, best_iter = loss
		loss = loss.mean()
		time_elapsed = time.time()-time_start
		if verbose > 0: print(". . .  loss=", np.round(loss, 3), "  time=", np.round(time_elapsed, 3))
		if "early_stopping_rounds" in fit_params:
			return {"params": params, "loss": loss, "status": STATUS_OK, "best_iter": best_iter, "time": time_elapsed}
		return {"params": params, "loss": loss, "status": STATUS_OK, "time": time_elapsed}
	print("Optimizing hyperparameters")
	space = {key: hp.choice(key, val) for key, val in param_grid.items()}
	trials = Trials()
	best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
	best = {"params": {param: param_grid[param][i] for param, i in best.items()}}
	best_trials = [record for record in trials.results if record["params"] == best["params"]]
	cols = ["loss", "best_iter", "time"] if "early_stopping_rounds" in fit_params else ["loss", "time"]
	for col in cols:
		best.update({col: np.array([record[col] for record in best_trials]).mean()})
	results = {param: [record["params"][param] for record in trials.results] for param in param_list}
	results.update({col: [np.round(record[col], 3) for record in trials.results] for col in cols})
	cols = param_list + cols
	results_df = pd.DataFrame.from_dict(results)[cols]
	times = {"mean_time": np.mean(results["time"]), "total_time": time.time()-time_0}
	args = {"model": model.__name__, "year": year, "cv": cv, "max_evals": max_evals, "n_cv_jobs": n_cv_jobs, "pca": pca}
	print("Saving results\n---\n")
	results_df.to_csv(out+"/results.csv", index=False)
	save_json(trials.results, out+"/results.json")
	save_json(init_params, out+"/init_params.json")
	save_json(param_grid, out+"/param_grid.json")
	save_json(fit_params, out+"/fit_params.json")
	save_json(args, out+"/args.json")
	save_json(best, out+"/best.json")
	save_json(times, out+"/times.json")


def model_run(model, year, init_params, fit_params={}, save_model=False, pca=False, out=None):
	print("\n---\nCalling model_run on year ", year, " data")
	time_0 = time.time()
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	out_preds = out
	out = out + "/" + str(year)
	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year, pca=pca)
	estimator = model(**init_params)
	print("Training")
	estimator.fit(x_train, y_train, **fit_params)
	del x_train; gc.collect()
	del y_train; gc.collect()
	if save_model:
		print("Saving model")
		joblib.dump(estimator, out+"/model.jl")
	print("Loading test data")
	x_test = load_test(year, pca=pca)
	if pca:
		x_test, update = x_test
		train_columns = ["pc"+str(i+1) for i in range(x_test.shape[1])]
	else:
		train_columns = load_train_columns(year)
		month_col_index = train_columns.index("month")
	print("Loading submission")
	if year == 2016:
		sub = pd.read_csv("data/sample_submission.csv")
	else:
		sub = pd.read_csv(out_preds+"/preds.csv")
	time_1 = time.time()
	print("Predicting")
	for month in [10, 11, 12]:
		print("\tmonth = ", month)
		if pca:
			x_test = x_test + update
		else:
			x_test[:,month_col_index] = month
		sub[str(year)+str(month)] = estimator.predict(x_test)
	time_2 = time.time()
	print("Saving results\n---\n")
	sub.to_csv(out_preds+"/preds.csv", index=False, float_format='%.4f')
	importances = pd.DataFrame.from_dict({"feature": train_columns, "importance": estimator.feature_importances_})[["feature", "importance"]]
	importances.sort_values(by="importance", axis=0, ascending=False, inplace=True)
	importances.to_csv(out+"/importances.csv", index=False)
	times = {"pred_time": time_2-time_1, "total_time": time.time()-time_0}
	args = {"model": model.__name__, "year": year, "pca": pca}
	fit_params.pop("callbacks", None)
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

