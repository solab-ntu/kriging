import numpy as np


class Kparam:

    """Kriging parameter
    Args:
    Attributes:
    """

    def __init__(self):
       
        self.inputs = None
        self.outputs = None
        self.method = None
        self.theta = None
        self.p = None
        self.nugget = None
        self.order = None
        self.lb = None
        self.ub = None
        self.beta = None
        self.sigma2 = None
        self.K = None
        self.mse = None

    def fit(self):
        pass

    def plot(self):
        pass
