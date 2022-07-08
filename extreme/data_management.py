from .distributions import Burr, GeneralizedHeavyTailed, Frechet, InverseGamma, Fisher, GPD, Student, NHW
import numpy as np
from pathlib import Path



dict_distributions = {"burr": Burr, "invgamma": InverseGamma, "frechet": Frechet, "fisher": Fisher, "gpd": GPD,
                      "student": Student, "generalized": GeneralizedHeavyTailed, "nhw": NHW}

def load_distribution(name_distribution):
    """load a distribution"""
    return dict_distributions[name_distribution]

def load_quantiles(distribution, params, n_data, rep=32, **kwargs):
    file = Path("data", distribution, "Xorder_{}_{}_ndata{}-rep{}.npy".format(distribution, params, n_data, rep))
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
            [x1, x2, y], X
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





