import pandas as pd
import scipy.stats
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri

from extreme.data_management import load_quantiles, DataSampler
import numpy as np
from pathlib import Path


list_estimators = ["W", "RW", "CW", "CH", "CHp", "CHps", "PRBp", "PRBps"]

# ==================================================
#                  Tail index estimator
# ==================================================

def hill(X, k_anchor):
    """

    Parameters
    ----------
    X : ndarray
        order statistics
    k : threshold
        int
        greater than 1

    Returns
    -------

    """
    X_in = X[-k_anchor:]
    X_kn = X[-(k_anchor+1)] * np.ones_like(X_in)

    return np.mean(np.log(X_in) - np.log(X_kn))


class TailIndexEstimator():
    def __init__(self, X_order):
        """
        Tail index estimators

        The class contains:
        - Hill (H) [1]
        - Corrected Hill (CH) [2]
        - (H_p) [3]
        - (CH_p) [4]
        - (CH_{p^star}) [5]
        - (PRB_P) [6]
        - (PRB_{p^star}) [7]

        Parameters
        ----------
        X_order : ndarray
            Order statistics X_{1,n} \leq ... \leq X_{n,n}

        References
        ----------

        Examples
        --------
        """
        # R settings
        rpy2.robjects.numpy2ri.activate()
        r = ro.r
        r['source']('extreme/revt.R')
        self.get_rho_beta = r.get_rho_beta
        self.l_run = r.lrun

        self.X_order = X_order
        self.n_data = X_order.shape[0]

        self.rho, self.beta = self.get_rho_beta(X_order)  # rho and beta estimated
        self.varphi = 1 - self.rho/2 - np.sqrt(np.square(1 - self.rho/2) - 0.5)
        self.k0 = self.get_k0()
        self.p_star = self.varphi / self.corrected_hill(self.k0)
        return

    def hill(self, k_anchor):
        """
        Hill estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        X_in = self.X_order[-k_anchor:]
        X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)  # take also the last point X_n,n
        return np.mean(np.log(X_in) - np.log(X_kn))

    def corrected_hill(self, k_anchor):
        """
        Corrected Hill estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma_hill = self.hill(k_anchor)
        return gamma_hill * (1 - (self.beta / (1 - self.rho) * np.power(self.n_data / k_anchor, self.rho)))

    def hill_p(self, k_anchor, p):
        """
        Redcued-bias H_p

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        p: float
            Tuning parameter

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        if p == 0.:
            gamma = self.hill(k_anchor)
        else:
            X_in = self.X_order[-k_anchor:]
            X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)
            gamma = (1 - np.power(np.mean(np.power(X_in / X_kn, p)), -1)) / p
        return gamma


    def corrected_hill_p(self, k_anchor, p=None):
        """
        Reduced-bias mean of order (CH_p)

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        if p is None:
            p = self.p_CH
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - p * gamma)) / (1 - self.rho - p * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def corrected_hill_ps(self, k_anchor):
        """
        Corrected Hill estimator with p^*

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.p_star * gamma)) / (1 - self.rho - self.p_star * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_p(self, k_anchor, p=None):
        """
        Partially reduced bias estimator

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        p: float or None (default None)
            Tuning parameter

        Returns
        -------
        gamma: float
            PRB_p estimator
        """
        if p is None:
            p = self.p_PRB
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_ps(self, k_anchor):
        """
        Partially reduced bias estimator with optimal p^*

        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        gamma: float
            Tail index estimator
        """
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def get_k0(self):
        """
        Estimated best intermediate sequence to choose the optimal value of p in PRB_{p^\star} and PRB_{p^\star}
        Returns
        -------

        """
        term1 = self.n_data - 1
        term2 = np.power(np.square(1 - self.rho) * np.power(self.n_data, -2*self.rho) / (-2*self.rho*np.square(self.beta)), 1/(1-2*self.rho))
        return int(np.minimum(term1, np.floor(term2) + 1))


# ==================================================
#                  Extreme quantile estimator
# ==================================================


def weissman(X_order, alpha, k_anchor):
    """
    Parameters
    ----------
    X_orders : order statistics
    alpha : extreme order
    k_anchor : anchor point

    Returns
    -------

    Maths
    ----
    X_{n-k, n}(k/np)^gamma_hill(k) with 0<p<1 and k\in{1,...,n-1}

    """
    gamma_hill = hill(X_order, k_anchor)
    n_data = X_order.shape[0]
    X_anchor = X_order[-k_anchor]
    return X_anchor * np.power(k_anchor/(alpha * n_data), gamma_hill)


class ExtremeQuantileEstimator(TailIndexEstimator):
    def __init__(self, X, alpha):
        """
        Extreme quantile estimators

        The class contains:
        - Weissman (H) [1]
        - Refined Weissman (RW) [2]
        - Corrected Weissman (CW) [3]
        - (CH) [4]
        - (CH_{p^star}) [5]
        - (PRB_P) [6]
        - (PRB_{p^star}) [7]

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Data X_1, ..., X_n
        alpha : float
            extreme quantile level
        """
        super(ExtremeQuantileEstimator, self).__init__(X)
        self.alpha = alpha
        self.dict_q_estimators = {"W": self.weissman, "RW": self.r_weissman, "CW": self.c_weissman,
                                  "CH": self.ch_weissman, "CHps": self.chps_weissman,
                                   "PRBps": self.prbps_weissman}
        self.dict_qp_estimators = {"CHp": self.chp_weissman, "PRBp": self.prbp_weissman}
        self.dict_quantile_estimators = {**self.dict_q_estimators, **self.dict_qp_estimators}

        self.p_CH = self.get_p(method="CHp")
        self.p_PRB = self.get_p(method="PRBp")
        return

    def weissman(self, k_anchor):
        """
        Weissman estimator (W)
        Parameters
        ----------
        k_anchor : int
            Intermediate sequence

        Returns
        -------
        Quantile estimator: float
        """
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.hill(k_anchor))

    def r_weissman(self, k_anchor):
        """Revisited Weissman (RW)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        k_prime = k_anchor * np.power((-self.rho * np.log(extrapolation_ratio)) / ((1-self.rho) * (1 - np.power(extrapolation_ratio, self.rho))), 1/self.rho)
        return X_anchor * np.power(extrapolation_ratio, self.hill(int(np.ceil(k_prime))))

    def c_weissman(self, k_anchor):
        """Corrected Weissman (CW)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio *
                                        np.exp(self.beta * np.power(self.n_data/k_anchor, self.rho)
                                               * (np.power(extrapolation_ratio, self.rho) - 1) / self.rho), self.corrected_hill(k_anchor))

    def ch_weissman(self, k_anchor):
        """Corrected-Hill Weissman (CH)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill(k_anchor))

    def chp_weissman(self, k_anchor, p=None):
        """Corrected-Hill with Mean-of-order-p Weissman (CHp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_p(k_anchor, p))

    def chps_weissman(self, k_anchor):
        """Corrected-Hill with Mean-of-order-p star (optimal) Weissman (CHps)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_ps(k_anchor))

    def prbp_weissman(self, k_anchor, p=None):
        """Partially Reduced-Bias mean-of-order-p Weissman (PRBp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_p(k_anchor, p))

    def prbps_weissman(self, k_anchor):
        """Partially Reduced-Bias mean-of-order-p star (optimal) Weissman (PRBPs)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_ps(k_anchor))

    def quantile_estimator(self, method, k_anchor):
        return self.dict_quantile_estimators[method](k_anchor)

    def get_k(self, x):
        """
        best k based on Algo 1 from Gomes, 2018
        Parameters
        ----------
        x : ndarray
            estimator (gamma or quantiles)

        Returns
        -------

        """
        x = np.log(x)  # convert to log as proposed in the article
        #STEP 2
        list_runsize = []
        k_minmax_list = []
        j = 0
        optimal=False
        while not optimal:
            x_rounded = np.around(x, j)
            if np.unique(x_rounded).shape[0] == x_rounded.shape[0]:  # if all uniques
                optimal = True
            else:
                j += 1
        # STEP 3
        k_min, k_max = self.run_size(x, j)
        k_minmax_list.append((k_min, k_max))
        list_runsize.append(k_max - k_min)
        largest_k_min, largest_kmax = k_minmax_list[np.argmax(list_runsize)]
        # STEP 4
        selected_x = x[int(largest_k_min) : int(largest_kmax+1)]
        new_q_rounded = np.around(selected_x, j+1)
        K_T = np.where(new_q_rounded == scipy.stats.mode(new_q_rounded)[0])
        # STEP 4.2
        bestK = int(np.median(K_T)+largest_k_min)  # \geq 0
        return bestK


    def get_p(self, method):
        """
        get best p and k based on Algo 2 from Gomes, 2018
        Parameters
        ----------
        method :

        Returns
        -------

        """
        # STEP 1
        xi_star = self.corrected_hill(self.k0)
        p_ell = np.arange(16)/(16*xi_star)

        #STEP 2
        list_runsize = []
        for ell in range(16):
            # STEP 2.1:  applied on log(.) as suggested in the paper
            quantiles = np.log([self.dict_qp_estimators[method](k_anchor=k_anchor, p=p_ell[ell])[0] for k_anchor in range(2, self.n_data)])
            # STEP 2.2: find the minimum j>=0 s.t all q_rounded are distinct
            j = 0
            optimal = False
            while not optimal:
                q_rounded = np.around(quantiles, j)
                if np.unique(q_rounded).shape[0] == q_rounded.shape[0]:  # if all uniques
                    optimal = True
                else:
                    j += 1
            # STEP 2.3
            k_min, k_max = self.longest_run(q_rounded, j)
            list_runsize.append(k_max - k_min)
        largest_runsize_idx = np.argmax(list_runsize)
        # STEP 3
        p = largest_runsize_idx / (16*xi_star)
        return p[0]

    # 
    
    @staticmethod
    def longest_run(x, j):
        """
        Compute the run size k_min and k_max

        Parameters
        ----------
        x : ndarray
        j: int
            decimal point + 1

        Returns
        -------
        k_min, k_max: int, int
        """
        x = x[~np.isnan(x)]  # remove nans
        x = x[~np.isinf(x)]  # remove inf
        mat = np.zeros(shape=(len(x), j + 1))
        for idx in range(len(x)):
            for val in range(j):
                # split the integer into array. Add "1"*(j+1) to avoid problem with numbers starting by 0
                mat[idx, val] = int(str(int(float('% .{}f'.format(j)%np.abs(x[idx]))*10**j) + + int("1"*(j+1)))[val])

        diff_mat = np.diff(mat, axis=1)  # diff in between columns
        list_k = np.count_nonzero(diff_mat == 0., axis=1)  # count number of zeros in columns
        return np.min(list_k), np.max(list_k)


def tree_k(x, a=None, c=None, return_var=False):
    """
    choice of the best k based on the dyadic decomposition.
    returns the Python index (starts at 0). Add 2 to get the order level.
    """
    if a is None:
        a = 13
    if c is None:
        c = int(3*x.shape[0] / 4)
    b = int((c+a)/2)

    list_var = []
    finish = False
    while not finish:
        if (b-a) < 2:
            finish = True
        else:
            v1 = np.var(x[a:b+1])
            v2 = np.var(x[b:c+1])
            if v1 < v2:  # left wins
                list_var.append(v1)
                c = b
            else:  # right wins
                list_var.append(v2)
                a = b
            b = int((c + a) / 2)
    if return_var:
        return b, np.mean(list_var)
    return b

def random_forest_k(x, n_forests, a=13/498, c=0.75, seed=42):
    """
    Algorithm to choose the intermediate sequence on a stable region given observations X_1,...,X_n
    Parameters
    ----------
    x : ndarray or list
        Observations
    n_forests : int
        number of forests in the algorithm
    seed : int
        Seed for PRGN

    Returns
    -------
    k : int
        selected anchor point (python indexing)
    """
    np.random.seed(seed)
    # a0 = 13
    # c0 = int(3 * x.shape[0] / 4)
    a0 = int(a * x.shape[0])
    c0 = int(c * x.shape[0])
    b0 = int((a0+c0)/2)
    list_k = []
    # print(a0,b0,c0)

    for i in range(n_forests):
        a = np.random.randint(a0, c0)
        c = np.random.randint(a+1, c0+1)
        # initial search in two parts
        # ---------------------------
        # a = np.random.randint(a0, b0)
        # c = np.random.randint(b0, c0+1)
        # ----------------------------

        list_k.append(tree_k(x, a, c))

    # return list_k[np.argmin(np.array(list_k)[:, 1])][0]
    # return int(np.median(np.array(list_k)[:, 0]))
    return int(np.median(np.array(list_k)))


def random_forest_k_j(X, n_forests, k_u=0.03, k_d=0.75, j_l=0.04, j_r=1, seed=42):
    np.random.seed(seed)
    n_rows = X.shape[0] - 1
    n_cols = X.shape[1] - 1

    k_u_0 = int(n_rows * k_u)
    k_d_0 = int(n_rows * k_d)
    j_l_0 = int(n_cols * j_l)
    j_r_0 = int(n_cols * j_r)

    list_k = []
    list_j = []

    for t in range(n_forests):
        j_l_t = np.random.randint(j_l_0, j_r_0)
        j_r_t = np.random.randint(j_l_t+1, j_r_0 + 1)
        k_d_t = np.random.randint(k_u_0, np.minimum(k_d_0, j_l_t))
        k_u_t = np.random.randint(k_u_0, k_d_t+1)
        # k_u_t = np.random.randint(k_u_0, k_d_0)
        # k_d_t = np.random.randint(k_u_t + 1, np.minimum(k_d_0 + 1, j_l_t))

        k_t, j_t = tree_2D(X, k_u_t, k_d_t, j_l_t, j_r_t)
        list_k.append(k_t)
        list_j.append(j_t)

    return int(np.median(list_k)), int(np.median(list_j))


def tree_2D(X, k_u, k_d, j_l, j_r):

    k_M = int((k_u+k_d)/2)
    j_M = int((j_l+j_r)/2)

    while (((k_M - k_u) > 1) or ((j_M - j_l) > 1)):
        var_ul = np.nanvar(X[k_u:k_M+1, j_l:j_M+1])
        var_ur = np.nanvar(X[k_u:k_M+1, j_M:j_r+1])
        var_dl = np.nanvar(X[k_M:k_d+1, j_l:j_M+1])
        var_dr = np.nanvar(X[k_M:k_d+1, j_M:j_r+1])

        list_variances = [var_ul, var_ur, var_dl, var_dr]
        if np.min(list_variances) == var_ul:
            k_d = k_M
            j_r = j_M
        elif np.min(list_variances) == var_ur:
            k_d = k_M
            j_l = j_M
        elif np.min(list_variances) == var_dl:
            k_u = k_M
            j_r = j_M
        else:
            
            k_u = k_M
            j_l = j_M

        k_M = int((k_u + k_d) / 2)
        j_M = int((j_l + j_r) / 2)


    return k_M, j_M




def evt_estimators(n_replications, n_data, distribution, params, metric="median", return_full=False):
    """
    Evaluation of extreme quantile estimators based on simulated heavy-tailed data

    Parameters
    ----------
    n_replications :
    n_data :
    distribution :
    params :
    n_quantile :
    return_full :

    Returns
    -------

    """
    dict_evt = {estimator: {_metric: {"series": [], "rmse_bestK": None, "q_bestK": [],
                            "bestK": []}for _metric in ["mean", "median"]} for estimator in list_estimators}

    pathdir = Path("ckpt", distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}_ndata{}.npy".format(n_replications, n_data)), allow_pickle=True)[()]
    except FileNotFoundError:
        # Estimators
        # -----------------
        for replication in range(1, n_replications + 1):  # for each replication
            print("rep ", replication)
            X_order = load_quantiles(distribution, params, n_data, rep=replication)  # replication order statistics X_1,n, ..., X_n,n
            dict_q = {estimator: [] for estimator in list_estimators}  # dict of quantiles
            evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

            for estimator in list_estimators:
                for anchor_point in anchor_points:  # compute all quantile estimators
                    dict_q[estimator].append(evt_estimators.quantile_estimator(k_anchor=anchor_point, method=estimator)[0])

                bestK = random_forest_k(np.array(dict_q[estimator]), 10000)   

                # MEAN
                dict_evt[estimator]["mean"]["series"].append(dict_q[estimator])
                dict_evt[estimator]["mean"]["q_bestK"].append(dict_q[estimator][int(bestK)])
                dict_evt[estimator]["mean"]["bestK"].append(bestK + 2)  # k \geq 2

                # MEDIAN
                dict_evt[estimator]["median"]["series"].append(dict_q[estimator])
                dict_evt[estimator]["median"]["q_bestK"].append(dict_q[estimator][int(bestK)])
                dict_evt[estimator]["median"]["bestK"].append(bestK + 2)  # k \geq 2

        # if not Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)).is_file():
        for estimator in list_estimators:
            # MEAN
            dict_evt[estimator]["mean"]["var"] = np.array(dict_evt[estimator]["mean"]["series"]).var(axis=0)
            dict_evt[estimator]["mean"]["rmse"] = ((np.array(dict_evt[estimator]["mean"]["series"]) / real_quantile - 1) ** 2).mean(axis=0)
            dict_evt[estimator]["mean"]["series"] = np.array(dict_evt[estimator]["mean"]["series"]).mean(axis=0)
            dict_evt[estimator]["mean"]["rmse_bestK"] = ((np.array(dict_evt[estimator]["mean"]["q_bestK"]) / real_quantile - 1) ** 2).mean()

            # MEDIAN
            dict_evt[estimator]["median"]["var"] = np.array(dict_evt[estimator]["median"]["series"]).var(axis=0)
            dict_evt[estimator]["median"]["rmse"] = np.median((np.array(dict_evt[estimator]["median"]["series"]) / real_quantile - 1) ** 2, axis=0)
            dict_evt[estimator]["median"]["series"] = np.median(dict_evt[estimator]["median"]["series"], axis=0)
            dict_evt[estimator]["median"]["rmse_bestK"] = np.median((np.array(dict_evt[estimator]["median"]["q_bestK"]) / real_quantile - 1) ** 2)

        np.save(Path(pathdir, "evt_estimators_rep{}_ndata{}.npy".format(n_replications, n_data)), dict_evt)

    if return_full:
        return dict_evt
    df = pd.DataFrame(columns=list_estimators, index=["RMSE"])
    for estimator in list_estimators:
        df.loc["RMSE", estimator] = dict_evt[estimator][metric]["rmse_bestK"]
    return df







