import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.001):
    return np.where(x < 0, alpha * x, x)

def de_leaky_relu(x, alpha=0.001):
    return np.where(x < 0, alpha, x)