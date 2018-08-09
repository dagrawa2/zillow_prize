import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

print("Loading data")
property_data = pd.read_csv('preprocessed/properties_2016.csv')
train_data = pd.read_csv('data/train_2016_v2.csv', parse_dates =["transactiondate"])

train_data['month'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).month)
#train_data['day'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).day)
#train_data['year'] = train_data['transactiondate'].apply(lambda x: (x.to_pydatetime()).year)
train_data.drop("transactiondate", axis=1, inplace=True)

print("Merging data")
merged_data = train_data.merge(property_data, on='parcelid', how='left')
merged_data.drop("parcelid", axis=1, inplace=True)

print("Cutting from double to single precision")
for c, dtype in zip(merged_data.columns, merged_data.dtypes):	
	if dtype == np.float64 or dtype == np.int64:		
		merged_data[c] = merged_data[c].astype(np.float32)

print("Saving merged data")
merged_data.to_csv("preprocessed/merged_2016.csv", index=False)

print("Done")
