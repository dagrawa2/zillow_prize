import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

N = 10

C_0 = np.random.uniform(-1, 1, size=(N, 2))
above_db = C_0[:,0]+C_0[:,1]>0
C_0[above_db] = -C_0[above_db]

C_1 = np.random.uniform(-1, 1, size=(N, 2))
below_db = C_1[:,0]+C_1[:,1]<0
C_1[below_db] = -C_1[below_db]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(C_0[:,0], C_0[:,1], color="blue", label="Class 0")
ax.scatter(C_1[:,0], C_1[:,1], color="red", label="Class 1")
ax.plot([-1, 1], [1, -1], color="black")
ax.set_xticklabels([]*len(ax.get_xticklabels()))
ax.set_yticklabels([]*len(ax.get_yticklabels()))
ax.legend()
fig.savefig("toy-1.png", bbox_inches='tight')

n_steps = 10
L = [-1+0.5/n_steps + i/n_steps for i in range(2*n_steps)]
x = []
for l in L:
	x.append(l)
	x.append(l)
x = [-1] + x + [1]

L = [-1 + i/n_steps for i in range(2*n_steps+1)]
L.reverse()
y = []
for l in L:
	y.append(l)
	y.append(l)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(C_0[:,0], C_0[:,1], color="blue", label="Class 0")
ax.scatter(C_1[:,0], C_1[:,1], color="red", label="Class 1")
ax.plot([-1, 1], [1, -1], color="black")
ax.plot(x, y, color="black", linestyle="dashed")
ax.set_xticklabels([]*len(ax.get_xticklabels()))
ax.set_yticklabels([]*len(ax.get_yticklabels()))
ax.legend()
fig.savefig("toy-2.png", bbox_inches='tight')
