import numpy as np


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
