import datetime
import gc
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

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
	print("\n---\nCalling gen_train on year ", year)
	year = str(year)
	print("Loading data")
	property_data = pd.read_csv('preprocessed/properties_'+year+'.csv')
	if year == "2016":
		train_data = pd.read_csv('data/train_'+year+'_v2.csv', parse_dates =["transactiondate"])
	else:
		train_data = pd.read_csv('data/train_'+year+'.csv', parse_dates =["transactiondate"])
	train_data['month'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).month)
#	train_data['day'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).day)
#	train_data['year'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).year)
	train_data.drop("transactiondate", axis=1, inplace=True)
	print("Merging")
	train_data = train_data.merge(property_data, on='parcelid', how='left')
	train_data.drop(['parcelid', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1, inplace=True)
	train_columns = list(train_data.columns)
	train_columns.remove('logerror')
	print("Saving training data\n---\n")
	train_data.to_csv("preprocessed/train_"+year+".csv", index=False)
	save_json(train_columns, "preprocessed/train_"+year+"_cols.json")

def load_train(year):
	if year == 0:
		boston = load_boston()
		return boston.data[:400], boston.target[:400]
	year = str(year)
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
	print("\n---\nCalling gen_test on year ", year)
	year = str(year)
	print("Loading data")
	train_columns = load_json("preprocessed/train_"+year+"_cols.json")
	property_data = pd.read_csv("preprocessed/properties_"+year+".csv")
	sample = pd.read_csv('data/sample_submission.csv')
	sample['parcelid'] = sample['ParcelId']
	print("Merging")
	df_test = sample.merge(property_data, on='parcelid', how='left')
	del property_data; gc.collect()
	del sample; gc.collect()
	df_test["month"] = 0
	x_test = df_test[train_columns]
	del df_test; gc.collect()
	print("Saving test data\n---\n")
	x_test.to_csv("preprocessed/test_"+year+".csv", index=False)

def load_test(year):
	if year == 0:
		boston = load_boston()
		return boston.data[400:], boston.target[400:]
	year = str(year)
	x_test = pd.read_csv("preprocessed/test_"+year+".csv")
	for c in x_test.dtypes[x_test.dtypes == object].index.values:
		x_test[c] = (x_test[c] == True)
	x_test = x_test.values.astype(np.float32, copy=False)
	return x_test

def load_train_columns(year):
	if year == 0:
		return ["col"+str(i+1) for i in range(13)]
	return load_json("preprocessed/train_"+str(year)+"_cols.json")

def load_train_types(year):
	if year == 0:
		return ["float" for i in range(13)]
	return load_json("preprocessed/train_"+str(year)+"_types.json")


def load_pretty_labels(year):
	if year == 0:
		return ["Column "+str(i+1) for i in range(13)]
	return load_json("preprocessed/train_"+str(year)+"_cols.json")
