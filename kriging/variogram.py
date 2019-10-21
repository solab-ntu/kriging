import numpy as np
from scipy.spatial.distance import pdist


class Variogram:

	def __init__(self, xs, ys, lb, ub):
		pass


if __name__ == "__main__":

    xs = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    ys = np.array([0, 1, 2, 1, 4, 7, 6]).reshape((-1, 1))

    lb = [0]
    ub = [6]

    variogram = Variogram(xs, ys, lb, ub)
