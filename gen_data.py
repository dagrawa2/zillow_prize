from utils import *

for year in [2016, 2017]:
	gen_train(year)
	gen_test(year)

print("Done")
