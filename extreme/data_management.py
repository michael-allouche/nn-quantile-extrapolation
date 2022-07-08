import numpy as np
import itertools

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import inv
from itertools import combinations

from .distributions import Burr, GeneralizedHeavyTailed, Frechet, InverseGamma, Fisher, GPD, Student, NHW
# from extreme.estimators import hill, random_forest_k
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pygmt
from itertools import product
from scipy.stats import median_absolute_deviation
from multiprocessing import Pool, Process, Manager


##########################################################################
#                             REAL DATA                                  #
##########################################################################

def get_intermediate_indices_2D(X):
    """
    Intemediate index sequences i=1,...,k-1 | k=2,...,n |
    Parameters
    ----------
    X :  ndarray
        series

    Returns
    -------
    list
        [(i,k,j)]
    """
    n_data = X.shape[0]
    triu_idx = np.triu_indices(n_data, 1)  # get the indexes of an upper triangular matrix (without the diag)
    i_idx = triu_idx[0] + 1  # i=1,...,n
    k_idx = triu_idx[1] + 1  # k=2 ,..., n
    list_indices = np.array(list(zip(i_idx, k_idx)))  # ((1,2), (1,3), ..., (1,K), ..., (K-1,K))
    return list_indices

def get_intermediate_indices_3D(X):
    """
    Intemediate index sequences i=1,...,k-1 | k=2,...,j-1 | j = 3, ..., n
    Parameters
    ----------
    X :  ndarray
        series

    Returns
    -------
    list
        [(i,k,j)]
    """
    n_data = X.shape[0]
    list_indices = []
    # Can dp better than for loop ? Combinations ? Triangular 3D mask ?
    for j in range(3, n_data+1):
        for k in range(2, j):
            for i in range(1, k):
                list_indices.append((i, k, j))
    return list_indices


def log_spacings(idx, X_order, a, b):
    """log X_{n-a+1, n} - log X_{n-b+1, n}"""
    return np.log(X_order[-idx[a]]) - np.log(X_order[-idx[b]])

def spacings(idx, X_order, a, b):
    """log X_{n-a+1, n} - log X_{n-b+1, n}"""
    return (X_order[-idx[a]]) - (X_order[-idx[b]])

def getKneighbors(data, n_knn):
    """
    Implement K nearest neighbors based on mahalanobis distance
    Parameters
    ----------
    data :
    n_knn :

    Returns
    -------

    """

    cov = np.cov(data.T)
    inv_covmat = inv(cov)

    list_distances = []
    n_data = data.shape[0]
    for i in range(n_data):  # for each observation
        dist = mahalanobis(x=data.iloc[i, :], data=data, inv_covmat=inv_covmat)
        list_distances.append(dist)

    indices = np.argsort(np.array(list_distances))
    distances = np.sort(np.array(list_distances))

    return indices[:, :n_knn], distances[:, :n_knn], inv_covmat


def mahalanobis(x, data=None, inv_covmat=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    X : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """

    if inv_covmat is None:
        cov = np.cov(data.T)
        inv_covmat = inv(cov)

    term1 = np.dot((x-data), inv_covmat)
    out = np.dot(term1, (x-data).T)

    return np.sqrt(out.diagonal())


def build_real_data(n_knn, n_kappa, data_type):
    """
    Build the read dataset of log-spacings modified
    Parameters
    ----------
    n_knn :  int
        nearest neighbors
    n_kappa :  int
        number of anchor points
    data_type : str
        {"LDreal", "Creal"}

    Returns
    -------
    dataset_train: ndarray
    dict_Xtrain: dict
    dict_y: dict
    dataset_test: ndarray
    dict_Xtest: dict

    """
    if data_type == "Creal":
        dict_indices = {"i": [], "k": []}
        dataset_train = {"x1": None, "x2": [], "y_lat": [], "y_long": [], "y_alt": [], "Z": []}  # init the dataset
        dataset_test = {"x1": None, "x2": [], "y_lat": [], "y_long": [], "y_alt": [], "Z": []}  # init the dataset
    elif data_type == "LDreal":
        dict_indices = {"i": [], "k": [], "j": []}
        dataset_train = {"x1": None, "x2": [], "x3": None, "Z": []}  # init the dataset
        dataset_test = {"x1": None, "x2": [], "x3": None, "Z": []}  # init the dataset

    dict_Xtest = {}
    dict_y = {}  # save the latitude/longitude for each ball

    df_rains = pd.read_csv(Path("data", "cevennes", "rains_cevennes.csv"), index_col=0)
    df_stations = pd.read_csv(Path("data", "cevennes", "stations_cevennes.csv"), index_col=0)

    # KNN estimator
    # y = df_stations[["lat", "long"]]
    # nearest_neighbors = NearestNeighbors(n_neighbors=n_knn).fit(y)  # compute the KNN for all points.
    y = df_stations[["lat", "long", "alt"]]  # conditional variable
    # nearest_neighbors = NearestNeighbors(n_neighbors=n_knn, algorithm='brute', metric="mahalanobis", metric_params={'VI': np.cov(y.T)}).fit(y)
    # distances_knn, indices_knn = nearest_neighbors.kneighbors(y)
    indices_knn, distances_knn, inv_covmat_knn = getKneighbors(data=y, n_knn=n_knn)
    # save the KNN information: inv covariance matrix, vector of medians and vector of mads
    median_knn = np.median(distances_knn, axis=1)
    mad_knn = median_absolute_deviation(distances_knn, axis=1)
    if data_type == "Creal":
        np.savez(Path("data", "Creal", "knn_infos-nknn{}-nkappa{}.npz".format(n_knn, n_kappa)),
                 inv_cov=inv_covmat_knn,
                 median = median_knn,
                 mad = mad_knn)


    # n_data = df_rains.shape[0]
    n_stations = df_rains.shape[1]
    n_observations = n_knn * df_rains.shape[0]
    dict_Xtrain = {}
    for i_station in range(n_stations):  # for all stations
        X = df_rains.iloc[:, indices_knn[i_station]].values.ravel()  # concatenante all values in the K-stations around the i_station
        X_order = np.unique(X[~np.isnan(X)])[-n_kappa:]  # remove nans and select the unique n_kappa greatest order statistics

        # train/test split the max observation around each station
        dict_Xtrain["ball_{}".format(i_station)] = X_order[:-1]  # wihtout the max order stats
        dict_Xtest["ball_{}".format(i_station)] = X_order[-1:]  # max order stats
        dict_y["ball_{}".format(i_station)] = (df_stations.iloc[i_station]["lat"],
                                               df_stations.iloc[i_station]["long"],
                                               df_stations.iloc[i_station]["alt"],
                                               df_stations.iloc[i_station]["y_merc"],
                                               df_stations.iloc[i_station]["x_merc"])

        if data_type == "Creal":
            list_indices = get_intermediate_indices_2D(X_order)
            mat_indices = np.array(list_indices)
            # Order statistics for the Conditional problem
            Z = np.apply_along_axis(log_spacings, 1, list_indices, X_order, 0, 1)  # spacings (i,k)
        elif data_type == "LDreal":
            list_indices = get_intermediate_indices_3D(X_order)
            mat_indices = np.array(list_indices)
            # Order statistics for the Dispersion-Location problem
            S_ik = np.apply_along_axis(spacings, 1, list_indices, X_order, 0, 1)  # spacings (i,k)
            S_kj = np.apply_along_axis(spacings, 1, list_indices, X_order, 1, 2)  # spacings (k,j)
            Z = S_ik / S_kj  # variable of interest
            dict_indices["j"].append(mat_indices[:, 2])

        dataset_train["Z"].append(Z)
        dict_indices["i"].append(mat_indices[:, 0])
        dict_indices["k"].append(mat_indices[:, 1])
        if data_type == "Creal":  # add the gps inputs
            dataset_train["y_lat"].append(df_stations.iloc[i_station]["lat"] * np.ones_like(Z))
            dataset_train["y_long"].append(df_stations.iloc[i_station]["long"] * np.ones_like(Z))
            dataset_train["y_alt"].append(df_stations.iloc[i_station]["alt"] * np.ones_like(Z))

    dataset_train["Z"] = np.concatenate(dataset_train["Z"]).reshape(-1, 1).astype('float32')
    dict_indices["i"] = np.concatenate(dict_indices["i"]).reshape(-1, 1).astype('int32')
    dict_indices["k"] = np.concatenate(dict_indices["k"]).reshape(-1, 1).astype('int32')
    dataset_train["x1"] = np.log(dict_indices["k"] / dict_indices["i"]).astype('float32')
    dataset_train["x2"] = np.log(n_observations / dict_indices["k"]).astype('float32')  # log(n/k)
    if data_type == "Creal":
        dataset_train["y_lat"] = np.concatenate(dataset_train["y_lat"]).reshape(-1, 1).astype('float32')
        dataset_train["y_long"] = np.concatenate(dataset_train["y_long"]).reshape(-1, 1).astype('float32')
        dataset_train["y_alt"] = np.concatenate(dataset_train["y_alt"]).reshape(-1, 1).astype('float32')
    if data_type == "LDreal":
        dict_indices["j"] = np.concatenate(dict_indices["j"]).reshape(-1, 1).astype('float32')
        dataset_train["x3"] = np.log(dict_indices["k"] / dict_indices["j"]).astype('float32')

    # build dataset test
    indices_test = np.where(dict_indices["i"] == 1)
    for key in dataset_train.keys():
        dataset_test[key] = dataset_train[key][indices_test].reshape(-1, 1)
        dataset_train[key] = np.delete(dataset_train[key], indices_test).reshape(-1,1)
    return dataset_train, dict_Xtrain, dict_y, dataset_test, dict_Xtest

def load_real_data(n_knn=50, n_kappa=100, data_type = "Creal", just_dictXy=False):
    """load real data from the Cevennes region: either from conditional (Creal) or from location-dispersion conditional (LDreal) model"""
    datatrain_filename = "datatrain_cevennes_nknn{}_nkappa{}.npz".format(n_knn, n_kappa)
    datatest_filename = "datatest_cevennes_nknn{}_nkappa{}.npz".format(n_knn, n_kappa)
    Xtrain_filename = "Xtrain_cevennes_nknn{}_nkappa{}.npy".format(n_knn, n_kappa)
    Xtest_filename = "Xtest_cevennes_nknn{}_nkappa{}.npy".format(n_knn, n_kappa)
    y_filename = "y_cevennes_nknn{}_nkappa{}.npy".format(n_knn, n_kappa)

    pathdir = Path("data", data_type)

    pathfile_datatrain = pathdir / datatrain_filename
    pathfile_datatest = pathdir / datatest_filename
    pathfile_Xtrain = pathdir / Xtrain_filename
    pathfile_Xtest = pathdir / Xtest_filename
    pathfile_y = pathdir / y_filename

    list_datatrain = []
    list_datatest = []
    try:
        if just_dictXy:
            return np.load(pathfile_Xtrain, allow_pickle=True).item(), np.load(pathfile_y, allow_pickle=True).item(), \
                   np.load(pathfile_Xtest, allow_pickle=True).item()

        data_train = np.load(pathfile_datatrain)
        data_test = np.load(pathfile_datatest)
        dict_Xtrain = np.load(pathfile_Xtrain, allow_pickle=True).item()
        dict_Xtest = np.load(pathfile_Xtest, allow_pickle=True).item()
        dict_y = np.load(pathfile_y, allow_pickle=True).item()

        for key in data_train.keys():
            list_datatrain.append(data_train[key])
            list_datatest.append(data_test[key])
        return np.concatenate(list_datatrain, axis=1), dict_Xtrain, dict_y, np.concatenate(list_datatest, axis=1), dict_Xtest

    except (FileExistsError, FileNotFoundError):
        data_train, dict_Xtrain, dict_y, data_test, dict_Xtest = build_real_data(n_knn=n_knn, n_kappa=n_kappa, data_type=data_type)

        np.save(pathfile_Xtrain, dict_Xtrain)
        np.save(pathfile_Xtest, dict_Xtest)
        np.save(pathfile_y, dict_y)
        np.savez(pathfile_datatrain, **data_train)
        np.savez(pathfile_datatest, **data_test)

        for key in data_train.keys():
            list_datatrain.append(data_train[key])
            list_datatest.append(data_test[key])

        if just_dictXy:
            return dict_Xtrain, dict_y, dict_Xtest

        return np.concatenate(list_datatrain, axis=1), dict_Xtrain, dict_y, np.concatenate(list_datatest, axis=1), dict_Xtest

def load_global_gps():
    """
    build global dataset with all points in the pygmt grid with high resolution.
    contains 11,598,961 points (longitude, latitude, elevation)
    """
    minlong, maxlong = 2.5, 6
    minlat, maxlat = 43.2, 45.5
    pathfile = Path("data", "Creal", "global_gps-{}-{}-{}-{}.npy".format(minlong, maxlong, minlat, maxlat))
    try:
        data = np.load(pathfile)
    except FileNotFoundError:
        grid = pygmt.datasets.load_earth_relief(resolution="03s", region=[minlong, maxlong, minlat, maxlat])
        long = grid.lon.to_numpy()
        lat = grid.lat.to_numpy()
        data = np.concatenate([np.array((list(product(lat, long)))), grid.data.flatten().reshape(-1, 1)],
                              axis=1)
        np.save(pathfile, data)
    return data


##########################################################################
#                             SIMULATED DATA                             #
##########################################################################

dict_distributions = {"burr": Burr, "invgamma": InverseGamma, "frechet": Frechet, "fisher": Fisher, "gpd": GPD,
                      "student": Student, "generalized": GeneralizedHeavyTailed, "nhw": NHW}

def load_distribution(name_distribution):
    """load a distribution"""
    return dict_distributions[name_distribution]

def load_quantiles(distribution, params, n_data, rep=32, **kwargs):
    file = Path("data", "sim", distribution, "Xorder_{}_{}_ndata{}-rep{}.npy".format(distribution, params, n_data, rep))
    try:
        if file.is_file():  # load if exists
            return np.load(file, allow_pickle=True)
        else:  # else simulate them
            data_sampler = DataSampler(distribution, params)
            return data_sampler.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1, 1)
    except OSError:  # if file not properly saved, delete it and save it again
        print("file", file, " removed")
        file.unlink()  # remove the file
        data_sampler = DataSampler(distribution, params)
        return data_sampler.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1, 1)


class DataSampler():
    def __init__(self, distribution, params, percentile=0, **kwargs):
        self.distribution = distribution
        self.params = params
        self.ht_dist = load_distribution(distribution)(**params)  # heavy-tailed distribution
        self.percentile = percentile

        self.pathdir_data = Path("data", self.distribution)
        if not self.pathdir_data.is_dir():
            self.pathdir_data.mkdir()

        return

    def load_simulated_data(self, n_data, rep=0, saved=False):
        """
        Simulate dataset (x,y,z) and the associated order statistics

        Parameters
        ----------
        n_data :int
            number of simulations
        rep :  int
            replication to fix the seed
        saved : str
            if True, save the the data

        Returns
        -------
        ndarray, ndarray
            [x, z, y], X
        """
        threshold_K = int(n_data - (n_data * self.percentile)) - 1  # threshold K < n (integer)

        data_filename = "xyz_{}_{}_ndata{}-rep{}.npz".format(self.distribution, self.params, n_data, rep)
        Xorder_filename = "Xorder_{}_{}_ndata{}-rep{}.npy".format(self.distribution, self.params, n_data, rep)

        pathfile_data = Path(self.pathdir_data , data_filename)
        pathfile_Xorder = Path(self.pathdir_data , Xorder_filename)

        if pathfile_data.is_file() and pathfile_Xorder.is_file():  # if file exists, load existing data
            data = np.load(pathfile_data)
            X_order = np.load(pathfile_Xorder)
            return np.concatenate([data["x1"], data["x2"], data["y"]], axis=1), X_order
        else:
            return self.build_simulated_data(n_data, threshold_K, rep, saved, pathfile_data, pathfile_Xorder)

    def build_simulated_data(self, n_data, threshold_K, rep=0, saved=False, pathfile_data=None, pathfile_Xorder=None):
        """compute log spacings"""
        triu_idx = np.triu_indices(threshold_K, 1)  # get the indexes of an upper triangular matrix (without the diag)
        i_idx = triu_idx[0] + 1   # i=1,...,K-1
        k_idx = triu_idx[1] + 1  # k=2 ,..., K<n
        list_indices = np.array(list(zip(i_idx, k_idx)))  # ((1,2), (1,3), ..., (1,K), ..., (K-1,K))
        X_order = self.simulate_quantiles(n_data, seed=rep, random=True).reshape(-1, 1)  # X_{1,n}, ..., X_{n,n} with u~U(0,1)

        def log_spacings(idx):
            """log X_{n-i+1, n} - log X_{n-k+1, n}"""
            return np.log(X_order[-idx[0]]) - np.log(X_order[-idx[1]])

        y = np.apply_along_axis(log_spacings, axis=1, arr=list_indices)
        x1 = np.float32(np.log(k_idx / i_idx).reshape(-1, 1))
        x2 = np.float32(np.log(n_data / k_idx).reshape(-1, 1))

        if saved:
            np.save(pathfile_Xorder, X_order)
            np.savez(pathfile_data, x1=x1, x2=x2, y=y)
        return np.concatenate([x1, x2, y], axis=1), X_order

    def simulate_quantiles(self, n_data, low_bound=0., high_bound=1., random=True, seed=32, **kwargs):
        """
        simulate from quantile function  q
        quantiles(random=False) or order statistics (random=True) from heavy-tailed distribution
        Parameters
        ----------
        n_data :
        low_bound :
        up_bound :
        random : bool
            if true: drawn u values from a uniform distribution, else from a linear grid
        kwargs :

        Returns
        -------

        """
        if random:
            np.random.seed(seed)
            u_values = np.random.uniform(low_bound, high_bound, size=(int(n_data), 1))  # sample from U( [0, 1) )
            quantiles = np.float32(self.ht_dist.ppf(u_values))
            return np.sort(quantiles, axis=0)  # sort the order statistics
        else:
            u_values = np.linspace(low_bound, high_bound, int(n_data)).reshape(-1,1)  # endpoint=False
            return np.float32(self.ht_dist.ppf(u_values))


    @staticmethod
    def split(arr, cond):
        """split an array given a condition"""
        return [arr[cond], arr[~cond]]





