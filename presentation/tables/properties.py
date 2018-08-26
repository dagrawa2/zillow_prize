import numpy as np
import pandas as pd

ncols = 6
nrows = 5
data = pd.read_csv("../../data/properties_2016.csv", nrows=nrows, usecols=list(range(ncols)), skiprows=list(range(1, 10001)))
data.insert(ncols, "\\cdots", ["\\cdots"]*nrows)

file = open("properties.tex", "w")
file.write("\\begin{center}\n")
file.write("\\begin{tabular}{|")
for _ in range(ncols+1):
	file.write("c|")
file.write("} \\hline\n")
for c in data.columns[:-1]:
	file.write(c+" & ")
file.write(data.columns[-1]+" \\\\ \\hline\n")

for _ in range(ncols):
	file.write("\\vdots & ")
file.write("\\ddots \\\\\n")

for i in range(nrows):
	for j in range(ncols):
		file.write(str(data.iloc[i, j])+" & ")
	file.write(str(data.iloc[i, -1])+" \\\\\n")

for _ in range(ncols):
	file.write("\\vdots & ")
file.write("\\ddots \\\\\n")

file.write("\\hline\n")
file.write("\\end{tabular}\n")
file.write("\\end{center}")

file.close()
