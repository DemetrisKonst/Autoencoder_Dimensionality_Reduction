import numpy as np

def manhattan(a, b):
    return np.linalg.norm(a-b, ord=1)

def euclidean(a, b):
    return np.linalg.norm(a-b, ord=2)
