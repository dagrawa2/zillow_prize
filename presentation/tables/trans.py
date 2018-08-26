import numpy as np
import pandas as pd

nrows = 5
data = pd.read_csv("../../data/train_2016_v2.csv", nrows=nrows, skiprows=list(range(1, 1001)))

file = open("trans.tex", "w")
file.write("\\begin{center}\n")
file.write("\\begin{tabular}{|")
for _ in range(len(data.columns)):
	file.write("c|")
file.write("} \\hline\n")
for c in data.columns[:-1]:
	file.write(c+" & ")
file.write(data.columns[-1]+" \\\\ \\hline\n")

for _ in range(len(data.columns)-1):
	file.write("\\vdots & ")
file.write("\\vdots \\\\\n")

for i in range(nrows):
	for j in range(len(data.columns)-1):
		file.write(str(data.iloc[i, j])+" & ")
	file.write(str(data.iloc[i, -1])+" \\\\\n")

for _ in range(len(data.columns)-1):
	file.write("\\vdots & ")
file.write("\\vdots \\\\\n")

file.write("\\hline\n")
file.write("\\end{tabular}\n")
file.write("\\end{center}")

file.close()
