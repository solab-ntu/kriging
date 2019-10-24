import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import direct


class Variogram:

    def __init__(self, xs: np.array, ys: np.array, lb: np.array, ub: np.array):
        self.xs = xs
        self.ys = ys
        self.lb = lb
        self.ub = ub
        self.normalized_xs = (self.xs - self.lb) / (self.ub - self.lb)
        self.cloud = Cloud(self.normalized_xs, self.ys)
        self.experimental = Experimental(self.cloud)

    def fit(self):
        pass

    def plot(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        self.cloud.plot()
        self.experimental.plot()
        ax.set_xlabel("$h\ \mathrm{(normalized)}$")
        ax.set_ylabel("$\gamma(h)$")
        ax.set_title("Variogram")
        plt.show()


class Cloud:

    def __init__(self, normalized_xs: np.array, ys: np.array):
        self.hs = pdist(normalized_xs, metric='euclidean') # normalized distance
        self.gammas = (pdist(ys, metric='euclidean')**2) / 2.0 # semi variogram
        self.hs_max = np.max(self.hs)

    def plot(self):
        fig = plt.gcf()
        ax = plt.gca()
        ax.scatter(self.hs, self.gammas, s=1, c="r")
        

class Experimental:

    def __init__(self, cloud: Cloud, linspace_num: int=50):
        self.cloud = cloud
        self.nums = np.linspace(start=0, stop=cloud.hs_max, num=linspace_num, endpoint=True)
        self.hs = [(i+j)/2.0 for i, j in zip(self.nums[:-1], self.nums[1:])]
        self.gammas = self._get_gammas()

    def _get_gammas(self):
        gammas = []
        for i, j in zip(self.nums[:-1], self.nums[1:]):
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
        ax.scatter(self.hs, self.gammas, s=20, facecolors='none', marker="^", edgecolors="b")


class Theoretical:

    def __init__(self, hs: np.array):
        self.hs = hs
        self.sigma2 = None
        self.theta = None
        self.p = 1.99
        self.nugget = 1e-12
        self.gammas = None
        self.R = None
    
    def set_gammas(self):
        pass


class SpatialCorrelationCoefficient:

    @staticmethod
    def dace_gaussian(hs: np.array, theta: float, p: float) -> np.array:
        pass

    @staticmethod
    def matern_linear(hs: np.array, theta: float) -> np.array:
        pass

    @staticmethod
    def matern_cubic(hs: np.array, theta: float) -> np.array:
        pass


if __name__ == "__main__":

    xs = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    ys = np.array([0, 1, 2, 1, 4, 7, 6]).reshape((-1, 1))

    lb = np.array([0])
    ub = np.array([6])

    variogram = Variogram(xs, ys, lb, ub)
    variogram.plot()
