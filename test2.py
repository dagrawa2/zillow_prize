import numpy as np
import pandas as pd

preds = pd.read_csv("results/preds.csv").values[:,1:]
print("preds.shape: ", preds.shape)
print("preds.mean: ", preds.mean())
print("avg preds.std: ", preds.std(1).mean())
