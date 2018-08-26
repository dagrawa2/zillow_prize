import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

N = 20
x = np.random.uniform(0, 1, size=N)
y = np.random.uniform(0, 1, size=N)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y, color="black")
ax.plot([0, 1], [0.45, 0.45], color="black", linestyle="dashed")
ax.plot([0, 1], [0.55, 0.55], color="black", linestyle="dashed")
ax.set_xlabel("Feature A")
ax.set_ylabel("All Other Features")
ax.set_xticklabels([]*len(ax.get_xticklabels()))
ax.set_yticklabels([]*len(ax.get_yticklabels()))
fig.savefig("alt.png", bbox_inches='tight')
