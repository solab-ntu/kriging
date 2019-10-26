import numpy as np
from scipy.spatial.distance import cdist


class SCC:

    """
    Spatial correlation coefficient function
    """

    @staticmethod
    def gaussian_dace(hs: np.array, theta: float, p: float) -> np.array:
        R = np.exp(-theta*(hs**p))
        return R

    @staticmethod
    def matern_linear(hs: np.array, theta: float) -> np.array:
        pass

    @staticmethod
    def matern_cubic(hs: np.array, theta: float) -> np.array:
        pass
    

def predict(xs_new: np.array, kparam: "kriging.Kparam") -> tuple:

    theta = kparam.variogram.theoretical.theta
    p = kparam.variogram.theoretical.p
    sigma2 = kparam.variogram.theoretical.sigma2

    lb = kparam.variogram.lb
    ub = kparam.variogram.ub
    xs = kparam.variogram.normalized_xs
    ys = kparam.variogram.ys
    n, d = xs.shape
    
    xs_new = (xs_new - lb) / (ub - lb)
    ys_new = [] # mean
    vs_new = [] # variance

    for x_new in xs_new:

        hs_new = cdist(xs, x_new.reshape((1,d)))
        rx = SCC.gaussian_dace(hs_new, theta, p)

        _ = np.concatenate((rx.T, np.ones((1, 1))), axis=1)
        K_lambda = np.dot(_, np.linalg.inv(kparam.K_mat))

        _ = np.concatenate((ys, np.ones((1, 1))), axis=0)
        y_new = np.dot(K_lambda, _)

        _ = np.concatenate((rx, np.ones((1, 1))), axis=0)
        v_new = sigma2 * (1 - np.dot(K_lambda, _))

        ys_new.append(y_new[0, 0])
        vs_new.append(v_new[0, 0])

    return ys_new, vs_new
