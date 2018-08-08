import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

print("Loading data")
merged_data = pd.read_csv("preprocessed/merged.csv")

x_train = merged_data
x_train.drop("logerror", axis=1, inplace=True)
y_train = merged_data['logerror'].values

del merged_data; gc.collect()

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

print("Splitting into training and validation")
split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['learning_rate'] = 0.002
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.5
params['num_leaves'] = 60
params['min_data'] = 500
params['min_hessian'] = 1

"""
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3
"""

watchlist = [d_valid]
clf = lgb.train(params, d_train, 500, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

print("Loading and preparing test set")
sample = pd.read_csv('sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(property_data, on='parcelid', how='left')
del sample; gc.collect()

x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Predicting")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})

sub = pd.read_csv('sample_submission.csv')
x_test["month"] = 10
sub["201610"] = clf.predict(x_test)
x_test["month"] = 11
sub["201611"] = clf.predict(x_test)
x_test["month"] = 12
sub["201612"] = clf.predict(x_test)

print("Saving predictions")
sub.to_csv('results/preds.csv', index=False, float_format='%.4f')
