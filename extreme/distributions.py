import numpy as np
from scipy import stats

class Frechet2OC():
    def __init__(self):
        self.cste_slowvar = None  # L_\infty
        self.evi = None  # extreme value index
        self.rho = None  # J order parameters
        return

    def cdf(self, x):
        raise ("No distribution called")

    def sf(self, x):
        """survival function """
        return 1 - self.cdf(x)

    def ppf(self, u):
        """quantile function"""
        raise ("No distribution called")

    def isf(self, u):
        """inverse survival function"""
        return self.ppf(1 - u)

    def tail_ppf(self, x):
        """tail quantile function U(x)=q(1-1/x)"""
        return self.isf(1/x)

    def norm_ppf(self, u):
        "quantile normalized X>=1"
        return self.isf((1 - u) * self.sf(1))

    def epsilon_func(self, x):
        """RV_rho from Karamata representation"""
        return x**self.sop * self.ell_slowvar(x)

    def L_slowvar(self, x):
        raise ("No distribution called")

    def ell_slowvar(self, x):
        raise ("No distribution called")



class Burr(Frechet2OC):
    def __init__(self, evi, rho):
        super(Burr, self).__init__()
        self.evi = evi
        self.rho = np.array(rho)

        self.cste_slowvar = 1
        return

    def cdf(self, x):
        return 1 - (1 + x ** (- self.rho / self.evi)) ** (1 / self.rho)

    def ppf(self, u):
        return (((1 - u) ** self.rho) - 1) ** (- self.evi / self.rho)

    def L_slowvar(self, x):
        return (1 - x**self.rho) ** (-self.evi / self.rho)

    def ell_slowvar(self, x):
        return self.evi / (1-x**self.rho)

class InverseGamma(Frechet2OC):
    def __init__(self, evi):
        super(InverseGamma, self).__init__()
        self.evi = evi
        self.rho = np.array(-self.evi)
        self.law = stats.invgamma(1/self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

class Frechet(Frechet2OC):
    def __init__(self, evi):
        super(Frechet, self).__init__()
        self.evi = evi
        self.rho = np.array([-1.])
        self.law = stats.invweibull(1 / self.evi)
        return

    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

class Fisher(Frechet2OC):
    def __init__(self, evi):
        super(Fisher, self).__init__()
        self.evi = evi
        self.rho = np.array([-2./self.evi])
        self.law = stats.f(1, 2/self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)


class GPD(Frechet2OC):
    def __init__(self, evi):
        super(GPD, self).__init__()
        self.evi = evi
        self.rho = np.array([-self.evi])
        self.law = stats.genpareto(self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)


class Student(Frechet2OC):
    def __init__(self, evi):
        super(Student, self).__init__()
        self.evi = evi
        self.rho = np.array([-2*self.evi])
        self.law = stats.t(1/self.evi)
        return
    def cdf(self, x):
        return 2 * self.law.cdf(x) - 1

    def ppf(self, u):
        return self.law.ppf((u+1)/2)

class NHW(Frechet2OC):
    def __init__(self, evi, rho):
        super(NHW, self).__init__()
        self.evi = evi
        self.rho = np.array(rho)
        
    def ppf(self, u):
        t = 1 / (1-u)
        A = self.rho * (t ** self.rho) * np.log(t) / 2
        return np.power(t, self.evi) * np.exp(A / self.rho)

class GeneralizedHeavyTailed():
    def __init__(self, q0, evi, scale, rho):
        self.q0 = q0
        self.evi = evi
        self.scale = np.array(scale).reshape(1, -1)
        self.rho = np.array(rho).reshape(1, -1) #.cumsum()
        return

    def ppf(self, u):
        """quantile function"""
        return self.q0 * np.power(1-u, -self.evi) * np.exp(np.sum(self.scale * (np.power(1-u, -self.rho) - 1), axis=1)).reshape(-1,1)

    def isf(self, u):
        """inverse survival function"""
        return self.ppf(1 - u)

    def tail_ppf(self, x):
        """tail quantile function U(x)=q(1-1/x)"""
        return self.isf(1/x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    q0 = 1.
    evi = 0.5
    rho = [-1, -2]  # \bar rho_j order parameter
    scale = [-1, -2]
    n_data = 100
    u = np.linspace(0, 1-1/n_data, n_data).reshape(-1, 1)

    ht = GeneralizedHeavyTailed(q0, evi, scale, rho)
    quantiles = ht.isf(u)
    plt.plot(u, quantiles)
    plt.show()

