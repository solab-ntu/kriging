import numpy as np
from scipy.spatial.distance import cdist

from .utilities import SCC


class Kparam:

    """
    Kriging equation parameters
    """

    def __init__(self, variogram: "kriging.Variogram"=None, likelihood: "kriging.Likelihood"=None):
        self.variogram = variogram
        self.likelihood = likelihood
        self.R_mat = self._get_R_mat()
        self.K_mat = self._get_K_mat()
        self.beta = self._get_beta()

    def _get_R_mat(self) -> np.array:
        normalized_xs = self.variogram.normalized_xs
        theta = self.variogram.theoretical.theta
        p = self.variogram.theoretical.p
        nugget = self.variogram.theoretical.nugget
        hs_mat = cdist(normalized_xs, normalized_xs)
        R = SCC.gaussian_dace(hs_mat.reshape((-1,1)), theta, p)
        R_mat = R.reshape(hs_mat.shape)
        R_mat = (1.0-nugget)*R_mat + nugget*np.eye(hs_mat.shape[0])
        return R_mat

    def _get_K_mat(self) -> np.array:
        K_mat = np.pad(self.R_mat, ((0,1), (0,1)), mode='constant', constant_values=1.0)
        K_mat[-1, -1] = 0.0
        return K_mat

    def _get_beta(self) -> np.array:
        ys = self.variogram.ys
        F = np.ones((1, ys.shape[0]))
        R_mat_inv = np.linalg.inv(self.R_mat)
        a = np.dot(F, R_mat_inv)
        b = np.dot(a, F.T)
        beta = np.dot(a, ys) / b
        return beta[0, 0]
