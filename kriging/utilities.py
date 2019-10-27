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
    

"""
The following code is the single-output version of predict() function:

ys_new = [] # means
vs_new = [] # variances

for nx_new in nxs_new:

    hs_new = cdist(nxs, nx_new.reshape((1,d))) # (n, 1)
    rx = SCC.gaussian_dace(hs_new, theta, p) # (n, 1)

    rxt_plus_one = np.concatenate((rx.T, np.ones((1,1))), axis=1) # (1, n+1)
    K_lambda = np.dot(rxt_plus_one, np.linalg.inv(kparam.K_mat)) # (1, n+1)*(n+1, n+1) = (1, n+1)

    ys_plus_one = np.concatenate((ys, np.ones((1,1))), axis=0) # (n+1, 1)
    y_new = np.matmul(K_lambda, ys_plus_one) # (1, n+1)*(n+1, 1) = (1, 1)

    rx_plus_one = np.concatenate((rx, np.ones((1,1))), axis=0) # (n+1, 1)
    v_new = sigma2 * (1 - np.dot(K_lambda, rx_plus_one)) # (1, n+1)*(n+1, 1) = (1, 1)

    ys_new.append(y_new[0,0])
    vs_new.append(v_new[0,0])

return np.array(ys_new), np.array(vs_new)
"""

def predict(xs_new: np.array, kparam: "kriging.Kparam") -> tuple:

    """
    Isotropic variogram ordinary kriging
    """

    theta = kparam.variogram.theoretical.theta
    p = kparam.variogram.theoretical.p
    sigma2 = kparam.variogram.theoretical.sigma2

    lb = kparam.variogram.lb
    ub = kparam.variogram.ub

    nxs = kparam.variogram.normalized_xs # (n, d)
    n, d = nxs.shape

    ys = kparam.variogram.ys # (n, 1)

    nxs_new = (xs_new - lb) / (ub - lb) # (m, d)
    m = nxs_new.shape[0]

    nxs_mat = np.tile(nxs, reps=(m,1,1)) # (m, n, d)

    nxs_new_mat = np.reshape(nxs_new, newshape=(m,1,d)) # (m, 1, d)
    nxs_new_mat = np.tile(nxs_new_mat, reps=(1,n,1)) # (m, n, d)

    hs_new_mat = np.linalg.norm(nxs_mat-nxs_new_mat, axis=2).reshape((m,n,1)) # (m, n, 1)
    
    rx = SCC.gaussian_dace(hs_new_mat, theta, p) # (m, n, 1)
    rx_plus_one = np.concatenate((rx, np.ones((m,1,1))), axis=1) # (m, n+1, 1)
    rxt = np.transpose(rx, axes=(0,2,1)) # (m, 1, n)
    rxt_plus_one = np.concatenate((rxt, np.ones((m,1,1))), axis=2) # (m, 1, n+1)

    K_mat_inv = np.linalg.inv(kparam.K_mat) # (n+1, n+1)
    K_mat_inv = np.tile(K_mat_inv, reps=(m,1,1)) # (m, n+1, n+1)
    K_lambda = np.matmul(rxt_plus_one, K_mat_inv) # (m, 1, n+1)*(m, n+1, n+1) = (m, 1, n+1)

    ys_mat = np.tile(ys, reps=(m,1,1)) # (m, n, 1)
    ys_mat_plus_one = np.concatenate((ys_mat, np.ones((m,1,1))), axis=1) # (m, n+1, 1)
    ys_new = np.matmul(K_lambda, ys_mat_plus_one) # (m, 1, n+1)*(m, n+1, 1) = (m, 1, 1)

    vs_new = sigma2 * (1 - np.matmul(K_lambda, rx_plus_one)) # (m, 1, n+1)*(m, n+1, 1) = (m, 1, 1)

    return ys_new.reshape((-1,1)), vs_new.reshape((-1,1))
