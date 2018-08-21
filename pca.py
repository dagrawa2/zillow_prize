import numpy as np
import pandas as pd
import explore_tools as et

out = "results/explore"

for year in [2017]:
	et.pca(year, n_components=25, out=out)

print("Done")
