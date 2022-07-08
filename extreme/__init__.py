# from .distributions import Burr
import numpy as np




def get_slope(x, y):
    """compute slope between two arrays x and y"""
    X = x - np.mean(x)
    Y = y - np.mean(y)
    return (X @ Y) / (X @ X)

