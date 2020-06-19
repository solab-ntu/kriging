import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

from .utilities import SCC, predict
from .kparam import Kparam


class Variogram:

    """
    Isotropic variogram
    """

    def __init__(self, xs: np.array, ys: np.array, lb: np.array, ub: np.array):
        self.xs = xs
        self.ys = ys
        self.lb = lb
        self.ub = ub
        self.normalized_xs = (self.xs - self.lb) / (self.ub - self.lb)
        self.cloud = Cloud(self.normalized_xs, self.ys)
        self.experimental = Experimental(self.cloud)
        self.theoretical = Theoretical(self.experimental)

    def plot(self, cloud_enable: bool=False):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        self.experimental.plot()
        self.theoretical.plot()
        if cloud_enable:
            self.cloud.plot()
        ax.set_xlabel("$h\ \mathrm{(normalized)}$")
        ax.set_ylabel("$\gamma(h)$")
        ax.set_title("Variogram")
        ax.legend()
        plt.show()


class Cloud:

    """
    Empirical variogram
    """

    def __init__(self, normalized_xs: np.array, ys: np.array):
        self.hs = pdist(normalized_xs, metric='euclidean') # normalized distance
        self.hs_min = np.min(self.hs)
        self.hs_max = np.max(self.hs)
        self.gammas = 0.5 * (pdist(ys, metric='euclidean')**2) # <== also call semivariogram ?

    def plot(self):
        fig = plt.gcf()
        ax = plt.gca()
        ax.scatter(self.hs, self.gammas, s=1, c="r", label='Empirical')


class Experimental:

    """
    Experimental variogram
    """

    def __init__(self, cloud: Cloud, linspace_num: int=50):
        self.cloud = cloud
        nums = np.linspace(start=self.cloud.hs_min, stop=self.cloud.hs_max, num=linspace_num, endpoint=True)
        self.hs = self._get_hs(nums)
        self.gammas = self._get_gammas(nums)

    def _get_hs(self, nums: np.array) -> list:
        hs = [(i+j)/2.0 for i, j in zip(nums[:-1], nums[1:])]
        return hs

    def _get_gammas(self, nums: np.array) -> list:
        gammas = []
        for i, j in zip(nums[:-1], nums[1:]):
            cloud_filter = np.logical_and(i < self.cloud.hs, self.cloud.hs <= j)
            count = np.count_nonzero(cloud_filter)
            gamma = np.nan
            if count > 0:
                total = np.sum(self.cloud.gammas[cloud_filter])
                gamma = total / count
            gammas.append(gamma)
        return gammas

    def plot(self):
        fig = plt.gcf()
        ax = plt.gca()
        ax.scatter(self.hs, self.gammas, s=20, facecolors='none', marker="^", edgecolors="b", label="Experimental")


class Theoretical:

    """
    Theoretical variogram
    """

    def __init__(self, experimental: Experimental):
        self.experimental = experimental
        self.sigma2 = None
        self.theta = None
        self.p = 1.99
        self.nugget = 1e-12

    def _get_gammas(self, hs: np.array) -> np.array:
        R = SCC.gaussian_dace(hs, self.theta, self.p)
        gammas = self.nugget + self.sigma2*(1.0-R)
        return gammas

    def fit(self):

        def _get_obj(x):
            self.sigma2, self.theta = x
            obj = 0
            for exp_h, exp_gamma in zip(self.experimental.hs, self.experimental.gammas):
                if not np.isnan(exp_gamma) and exp_h < 1.0: # dont fit long-distance h, filter it by exp_h < 1.0
                    obj += (self._get_gammas(np.array(exp_h)) - exp_gamma)**2
            return obj

        b_sigma2 = (0.01, np.nanmax(self.experimental.gammas)*0.5)  # <=== limit sigma2
        b_theta = (0.01, 1000)
        res = differential_evolution(func=_get_obj, bounds=(b_sigma2, b_theta))
        self.sigma2, self.theta = res.x

    def plot(self):
        fig = plt.gcf()
        ax = plt.gca()
        hs = np.linspace(start=self.experimental.hs[0], stop=self.experimental.hs[-1], num=20, endpoint=True)
        gammas = self._get_gammas(hs)
        ax.plot(hs, gammas, label="Theoretical")

    def __repr__(self):
        return "sigma2 = {0:.3f}\ntheta = {1:.3f}\np = {2:.3f}\nnugget = {3:.3f}".format(self.sigma2, self.theta, self.p, self.nugget)
