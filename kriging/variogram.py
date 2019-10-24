import numpy as np
from scipy.spatial.distance import pdist


class VariogramFit:

    def __init__(self, xs: np.array, ys: np.array, lb: np.array, ub: np.array):
        self.xs = xs
        self.ys = ys
        self.lb = lb
        self.ub = ub
        self.normalized_xs = (self.xs - self.lb) / (self.ub - self.lb)
        self.cloud = VariogramCloud(self.normalized_xs, self.ys)
        self.experimental = VariogramExperimental(self.cloud)


class VariogramCloud:

    def __init__(self, normalized_xs: np.array, ys: np.array):
        self.hs = pdist(normalized_xs, metric='euclidean')
        self.2gammas = pdist(ys, metric='euclidean')**2
        self.h_max = np.max(self.hs)


class VariogramExperimental:

    def __init__(self, cloud: VariogramCloud):
        section_count = 50


if __name__ == "__main__":

    xs = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    ys = np.array([0, 1, 2, 1, 4, 7, 6]).reshape((-1, 1))

    lb = np.array([0])
    ub = np.array([6])

    variogram = Variogram(xs, ys, lb, ub)
    print(123)