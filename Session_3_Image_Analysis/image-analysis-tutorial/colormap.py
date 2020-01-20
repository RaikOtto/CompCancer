import numpy as np
from matplotlib.colors import ListedColormap

N = 256
vals = np.ones((N, 4))
vals[0, 0] = 0
vals[0, 1] = 0
vals[0, 2] = 0
for n in range(1, N):
    vals[n, 0] = np.random.rand()
    vals[n, 1] = np.random.rand()
    vals[n, 2] = np.random.rand()
rand_cmap = ListedColormap(vals)
