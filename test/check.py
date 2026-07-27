import numpy as np

X = np.load("datasets/sequences/X.npy")

print("NaN :", np.isnan(X).sum())
print("Inf :", np.isinf(X).sum())
print("Max :", X.max())
print("Min :", X.min())