import numpy as np
import pandas as pd
import explore_tools as et

year = 2017
out = "results/explore_"+str(year)

#et.explained_variance_ratio(year, out=out)
#et.pca(year, n_components=25, out=out)
#et.corrs(year, out=out)

#et.plot_evr(year, max_k=10, out=out)
et.plot_cross_sections(year, absolute=True, out=out)
#et.plot_corrs(year, out=out)

print("Done")
