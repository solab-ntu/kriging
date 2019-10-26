import numpy as np

import kriging

if __name__ == "__main__":

    xs = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    ys = np.array([0, 1, 2, 1, 4, 7, 6]).reshape((-1, 1))

    lb = np.array([0])
    ub = np.array([6])

    variogram = kriging.Variogram(xs, ys, lb, ub)
    variogram.theoretical.fit()
    # variogram.plot()

    kparam = kriging.Kparam(variogram)
    xs_new = np.array([1, 2.5, 5.5]).reshape((-1, 1))
    ys_new = kriging.predict(xs_new, kparam)
