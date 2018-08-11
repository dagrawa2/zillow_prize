import numpy as np
import pandas as pd

prop_2016 = pd.read_csv("data/properties_2016.csv", usecols=["parcelid"]).values.reshape((-1))
prop_2017 = pd.read_csv("data/properties_2017.csv", usecols=["parcelid"]).values.reshape((-1))

print("prop_2016 len: ", len(prop_2016))
print("prop_2017 len: ", len(prop_2017))
common = np.intersect1d(prop_2016, prop_2017, assume_unique=True)
print("common len: ", len(common))

print("\n")

train_2016 = pd.read_csv("data/train_2016_v2.csv", usecols=["parcelid"]).values.reshape((-1))
train_2017 = pd.read_csv("data/train_2017.csv", usecols=["parcelid"]).values.reshape((-1))

print("train_2016 len: ", len(train_2016))
print("train_2017 len: ", len(train_2017))

train_2016_unique = np.unique(train_2016)
train_2017_unique = np.unique(train_2017)

print("train_2016_unique len: ", len(train_2016_unique))
print("train_2017_unique len: ", len(train_2017_unique))

common = np.intersect1d(train_2016_unique, train_2017_unique, assume_unique=True)
common_2016 = np.intersect1d(prop_2016, train_2016_unique, assume_unique=True)
common_2017 = np.intersect1d(prop_2017, train_2017_unique, assume_unique=True)

print("train_2016 train_2017 common len: ", len(common))
print("prop_2016 train_2016 common len: ", len(common_2016))
print("prop_2017 train_2017 common len: ", len(common_2017))

