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

    """
    Isotropic variogram kriging
    """

    theta = kparam.variogram.theoretical.theta
    p = kparam.variogram.theoretical.p
    sigma2 = kparam.variogram.theoretical.sigma2

    lb = kparam.variogram.lb
    ub = kparam.variogram.ub
    nxs = kparam.variogram.normalized_xs
    ys = kparam.variogram.ys
    nxs_new = (xs_new - lb) / (ub - lb)

    n, d = nxs.shape
    m = nxs_new.shape[0]

    ys_new = [] # means
    vs_new = [] # variances

    for nx_new in nxs_new:

        hs_new = cdist(nxs, nx_new.reshape((1,d))) # (n, 1)
        rx = SCC.gaussian_dace(hs_new, theta, p) # (n, 1)

        _ = np.concatenate((rx.T, np.ones((1,1))), axis=1) # (1, n+1)
        K_lambda = np.dot(_, np.linalg.inv(kparam.K_mat)) # (1, n+1)*(n+1, n+1) = (1, n+1)

        _ = np.concatenate((ys, np.ones((1,1))), axis=0) # (n+1, 1)
        y_new = np.dot(K_lambda, _) # (1, n+1)*(n+1, 1) = (1, 1)

        _ = np.concatenate((rx, np.ones((1,1))), axis=0) # (n+1, 1)
        v_new = sigma2 * (1 - np.dot(K_lambda, _)) # (1, n+1)*(n+1, 1) = (1, 1)

        ys_new.append(y_new[0,0])
        vs_new.append(v_new[0,0])

    return np.array(ys_new), np.array(vs_new)
