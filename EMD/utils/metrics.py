from math import sqrt
import numpy as np

def manhattan(a, b):
    return np.linalg.norm(a-b, ord=1)

def euclidean(a, b):
    xa, ya = a
    xb, yb = b
    dist = sqrt((xa-xb)**2 + (ya-yb)**2)
    return dist
