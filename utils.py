import datetime
import gc
import json
import pickle
import pandas as pd
import numpy as np

def save_json(D, path):
	with open(path, "w") as f:
		json.dump(D, f, indent=2)

def load_json(path):
	with open(path, "r") as f:
		D = json.load(f)
	return D

def save_pickle(D, path):
	with open(path, "wb") as f:
		pickle.dump(D, f)

def load_pickle(path):
	with open(path, "rb") as f:
		D = pickle.load(f)
	return D

	

def gen_train(year):
	property_data = pd.read_csv('preprocessed/properties_'+year+'.csv')
	if year == 2016: year = year += "_v2"
	train_data = pd.read_csv('data/train_'+year+'.csv', parse_dates =["transactiondate"])
	train_data['month'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).month)
#	train_data['day'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).day)
#	train_data['year'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).year)
	train_data.drop("transactiondate", axis=1, inplace=True)
	train_data = train_data.merge(property_data, on='parcelid', how='left')
	train_data.drop(['parcelid', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1, inplace=True)
	train_columns = list(train_data.columns)
	train_columns.remove('logerror')
	train_data.to_csv("preprocessed/train_"+year+"_csv", index=False)
	save_pickle(train_columns, "preprocessed/train_2016_cols.list")

def load_train(year):
	x_train = pd.read_csv("preprocessed/train_"+year+".csv")
	for c, dtype in zip(x_train.columns, x_train.dtypes):	
		if dtype == np.float64 or dtype == np.int64:		
			x_train[c] = x_train[c].astype(np.float32)
	y_train = x_train['logerror'].values
	x_train.drop('logerror', axis=1, inplace=True)
	for c in x_train.dtypes[x_train.dtypes == object].index.values:
		x_train[c] = (x_train[c] == True)
	x_train = x_train.values.astype(np.float32, copy=False)
	return x_train, y_train

def gen_test(year):
	train_columns = load_pickle("preprocessed/train_"+year+"_cols.list")
	property_data = pd.read_csv("preprocessed/properties_"+year+".csv")
	sample = pd.read_csv('data/sample_submission.csv')
	sample['parcelid'] = sample['ParcelId']
	df_test = sample.merge(property_data, on='parcelid', how='left')
	del property_data; gc.collect()
	del sample; gc.collect()
	df_test["month"] = 0
	x_test = df_test[train_columns]
	del df_test; gc.collect()
	x_test.to_csv("preprocessed/test_+"year+".csv", index=False)

def load_test(year)
	x_test = pd.read_csv("preprocessed/test_"+year+"_.csv")
	for c in x_test.dtypes[x_test.dtypes == object].index.values:
		x_test[c] = (x_test[c] == True)
	x_test = x_test.values.astype(np.float32, copy=False)
	return x_test
