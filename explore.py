import numpy as np
import pandas as pd
import explore_tools as et

out = "results/explore"

for year in [2016, 2017]:
	et.fraction_missing(year, out=out)
	et.explained_variance_ratio(year, out=out)
	et.corrs(year, absolute=False, out=out)
	et.corrs(year, absolute=True, out=out)

	et.plot_fraction_missing(year, n_features=10, out=out)
	et.plot_evr(year, threshold=0.95, out=out)
	et.plot_cross_sections(year, absolute=False, out=out)
	et.plot_cross_sections(year, absolute=True, out=out)
	et.plot_corrs(year, out=out)
	et.plot_transactions_per_month(year, out=out)

print("Done")
