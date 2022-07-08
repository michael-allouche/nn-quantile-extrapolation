import pandas as pd
import numpy as np
from scipy import stats


def compute_criteria(X, criteria):
    def _variance():
        """variance"""
        return np.var(X)
    def _r_variance():
        return _variance() / np.square(np.mean(X))

    def _mad():
        """median absolute deviation"""
        return stats.median_abs_deviation(X)
    def _r_mad():
        return _mad() / np.median(X)

    def _aad():
        """absolute average deviation"""
        return pd.Series(X).mad()

    def _r_aad():
        return _aad() / np.mean(X)


    dict_functions = {"variance": _variance, "r_variance": _r_variance,
                      "mad": _mad, "r_mad": _r_mad,
                      "aad": _aad, "r_aad": _r_aad}
    return dict_functions[criteria]()

