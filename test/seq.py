import numpy as np

y = np.load("datasets/sequences/y.npy")

print(np.unique(y, return_counts=True))
