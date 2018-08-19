import gc
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.preprocessing import StandardScaler
from utils import *

class MidpointNormalize(Normalize):

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def explained_variance_ratio(year, out=None):
	if year == 0:
		print("\n---\nCalling explained_variance_ratio on Boston data")
	else:
		print("\n---\nCalling explained_variance_ratio on ", year, " data")
#	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, _ = load_train(year)
	print("Standardizing")
	x_train = (x_train - np.mean(x_train, axis=0, keepdims=True))/np.std(x_train, axis=0, keepdims=True)
	print("Computing singular values")
	s = np.linalg.svd(x_train, compute_uv=False)
	print("Computing ratios")
	eigs = s**2
	ratios = np.cumsum(eigs)/np.sum(eigs)
	results = {"k": range(1, len(ratios)+1), "ratio": list(ratios)}
	results = pd.DataFrame.from_dict(results)[["k", "ratio"]]
	print("Saving results\n---\n")
	results.to_csv(out+"/evr.csv", index=False)

def pca(year, n_components=1, out=None):
	if year == 0:
		print("\n---\nCalling pca on Boston data")
	else:
		print("\n---\nCalling pca on ", year, " data")
#	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, _ = load_train(year)
	print("Performing PCA")
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	_, _, V = np.linalg.svd(x_train)
	P = V[:n_components].T.dot(V[:n_components])
	x_train = x_train.dot(P)
	print("Saving projected training data")
	np.save("preprocessed/x_train_"+str(year)+"_pca_dim"+str(n_components)+".npy", x_train)
	del x_train; gc.collect()
	del V; gc.collect()
	print("Loading test data")
	x_test, _ = load_test(year)
	print("Projecting test data")
	x_test = scaler.transform(x_test)
	x_test = x_test.dot(P)
	print("Saving projected test data")
	np.save("preprocessed/x_test_"+str(year)+"_pca_dim"+str(n_components)+".npy", x_test)

def corrs(year, out=None):
	if year == 0:
		print("\n---\nCalling corrs on Boston data")
	else:
		print("\n---\nCalling cors on ", year, " data")
#	os.system("if [ ! -d "+out+" ]; then mkdir "+out+"; fi")
	print("Loading training data")
	x_train, y_train = load_train(year)
	train_columns = load_train_columns(year)
	data = np.concatenate((x_train, y_train.reshape((-1, 1))), axis=1)
	del x_train; gc.collect()
	del y_train; gc.collect()
	print("Computing correlations")
	pearson = np.corrcoef(data, rowvar=False)
	spearman, _ = sp_stats.spearmanr(data)
	del data; gc.collect()
	print("Generating CSV formats")
	corrs_with_y = {"feature": train_columns, "pearson": pearson[:-1,-1], "spearman": spearman[:-1,-1]}
	corrs_x = {"feature_1": [], "feature_2": [], "pearson": [], "spearman": []}
	for i, feature_1 in enumerate(train_columns[:-1]):
		for j, feature_2 in enumerate(train_columns[i+1:]):
			corrs_x["feature_1"].append(feature_1)
			corrs_x["feature_2"].append(feature_2)
			corrs_x["pearson"].append(pearson[i,i+1+j])
			corrs_x["spearman"].append(spearman[i,i+1+j])
	corrs_with_y = pd.DataFrame.from_dict(corrs_with_y)[["feature", "pearson", "spearman"]]
	corrs_x = pd.DataFrame.from_dict(corrs_x)[["feature_1", "feature_2", "pearson", "spearman"]]
	print("Generating sorted data")
	corrs_with_y_pearson_sorted = corrs_with_y[["feature", "pearson"]].sort_values(by="pearson", axis=0, ascending=False)
	corrs_with_y_spearman_sorted = corrs_with_y[["feature", "spearman"]].sort_values(by="spearman", axis=0, ascending=False)
	corrs_x_pearson_sorted = corrs_x[["feature_1", "feature_2", "pearson"]].sort_values(by="pearson", axis=0, ascending=False)
	corrs_x_spearman_sorted = corrs_x[["feature_1", "feature_2", "spearman"]].sort_values(by="spearman", axis=0, ascending=False)
	print("Saving results\n---\n")
	np.save(out+"/pearson.npy", pearson)
	np.save(out+"/spearman.npy", spearman)
	corrs_with_y.to_csv(out+"/corrs_with_y.csv", index=False)
	corrs_x.to_csv(out+"/corrs_x.csv", index=False)
	corrs_with_y_pearson_sorted.to_csv(out+"/corrs_with_y_pearson_sorted.csv", index=False)
	corrs_with_y_spearman_sorted.to_csv(out+"/corrs_with_y_spearman_sorted.csv", index=False)
	corrs_x_pearson_sorted.to_csv(out+"/corrs_x_pearson_sorted.csv", index=False)
	corrs_x_spearman_sorted.to_csv(out+"/corrs_x_spearman_sorted.csv", index=False)

def plot_evr(year, max_k=None, out=None):
	print("\n---\nCalling plot_ever on ", out, "/evr.csv")
#	os.system("if [ ! -d "+out+"/plots ]; then mkdir "+out+"/plots; fi")
	print("Loading data")
	data = pd.read_csv(out+"/evr.csv").values[:max_k]
	print("Plotting")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(data[:,0], data[:,1], color="black")
	ax.set_title("Explained variance ratio for "+str(year)+"data")
	ax.set_xlabel("Number of components")
	ax.set_ylabel("Ratio")
	print("Saving plot\n---\n")
	fig.savefig(out+"/plots/evr.png", bbox_inches='tight')

def plot_transactions_per_month(year, out=None):
	print("\n---\nCalling plot_transactions_per_month on ", year, " data")
#	os.system("if [ ! -d "+out+"/plots ]; then mkdir "+out+"/plots; fi")
	print("Loading month data")
	train_columns = load_train_columns(year)
	months, _ = load_train(year)
	months = months[:,train_columns.index("month")]
	months, counts = np.unique(months, return_counts=True)
	print("Plotting")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.bar(months, counts, width=0.2, color="black")
	ax.set_title("Number of transactions per month in "+str(year))
	ax.set_xlabel("Month")
	ax.set_ylabel("Transactions")
	ax.set_xticks(months)
#ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
#ax.set_ylim()
	print("Saving plot\n---\n")
	fig.savefig(out+"/plots/transactions.png", bbox_inches='tight')


def plot_cross_sections(year, absolute=False, out=None):
	if year == 0:
		print("\n---\nCalling plot_cross_sections on Boston data")
	else:
		print("\n---\nCalling plot_cross_sections on ", year, " data")
#	os.system("if [ ! -d "+out+"/plots ]; then mkdir "+out+"/plots; fi")
	print("Loading training data")
	x_train, y_train = load_train(year)
	train_columns = load_train_columns(year)
	pretty_labels = load_pretty_labels(year)
	abs_label = ""
	abs_filename = ""
	if absolute:
		y_train = np.abs(y_train)
		abs_label = "Absolute "
		abs_filename = "abs-"
	n_features = x_train.shape[1]
	for i in range(n_features):
		print("Plotting cross section ", i+1, " of ", n_features)
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.scatter(x_train[:,i], y_train, color="black")
		ax.set_title("2016 data "+abs_label+"cross section "+str(i+1))
		ax.set_xlabel(pretty_labels[i])
		ax.set_ylabel(abs_label+"log error")
		print("Saving plot")
		fig.savefig(out+"/plots/cs-"+abs_filename+train_columns[i]+".png", bbox_inches='tight')
	print("---\n")

def plot_corrs(year, out=None):
	if year == 0:
		print("\n---\nCalling plot_corrs on Boston data")
	else:
		print("\n---\nCalling plot_corrs on ", year, " data")
#	os.system("if [ ! -d "+out+"/plots ]; then mkdir "+out+"/plots; fi")
	data = np.abs( pd.read_csv(out+"/corrs_with_y.csv", usecols=["pearson", "spearman"]).values )
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(data[:,0], data[:,1], color="black")
	ax.set_title("Absolute correlation of each feature with target on "+str(year)+" data")
	ax.set_xlabel("Pearson")
	ax.set_ylabel("Spearman")
	print("Saving plot")
	fig.savefig(out+"/plots/corrs_with_y.png", box_inches='tight')
	data = np.abs( pd.read_csv(out+"/corrs_x.csv", usecols=["pearson", "spearman"]).values )
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(data[:,0], data[:,1], color="black")
	ax.set_title("Absolute correlation between features on "+str(year)+" data")
	ax.set_xlabel("Pearson")
	ax.set_ylabel("Spearman")
	print("Saving plot")
	fig.savefig(out+"/plots/corrs_x.png", bbox_inches='tight')
	corrs = np.abs( np.load(out+"/spearman.npy") )
	corrs_with_y = corrs[:-1,-1]
	corrs_x = corrs[:-1,:-1]
	inds = np.argsort(corrs_with_y)
	corrs_with_y = corrs_with_y[inds]
	corrs_x = corrs_x[inds,inds]
	x = np.repeat(corrs_with_y.reshape((1, -1)), repeats=len(corrs_with_y), axis=0)
	y = np.flip(x.T, axis=0)
	z = np.flip(corrs_x, axis=0)
	del corrs; del corrs_with_y; del corrs_x; del inds; gc.collect()
	vmin, midpoint = np.min(z), np.mean(z)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.pcolormesh(x, y, z, cmap=plt.cm.hot, norm=MidpointNormalize(vmin=vmin, midpoint=midpoint))
	ax.colorbar()
	ax.set_title("Absolute Spearman correlations of features over correlations with target on "+str(year)+" data")
	ax.set_xlabel("Feature 1 correlation")
	ax.set_ylabel("Feature 2 correlation")
	print("Saving plot")
	fig.savefig(out+"/plots/corrs_color.png", bbox_inches='tight')
