"""
1-d fitting nugget
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

import kriging

def _get_obj(x: tuple) -> float:
    criterion = 0.68
    xs = variogram.xs
    ys = variogram.ys
    total = xs.shape[0]
    count = 0
    variogram.theoretical.nugget = x[0]
    kparam = kriging.Kparam(variogram=variogram)
    ys_new, vs_new = kriging.predict(xs, kparam)
    for y, yn, vn in zip(ys, ys_new, vs_new):
        if yn-vn < y < yn+vn:
            count += 1
    ratio = count / total
    obj = x[0] if ratio > criterion else 1.0
    return obj

if __name__ == "__main__":

    xs = list()
    mus = list()
    ys = list()
    ss = list()

    for i in np.linspace(start=0, stop=10, num=50, endpoint=True):
        count = np.random.randint(low=1, high=20)
        # sigma = np.random.rand(1)[0]*1.5
        sigma = 1.5
        x = i
        mu = np.sin(i) + 1.0*i
        mus.append(mu)
        ss.append(sigma)
        for _ in range(count):
            y = mu + np.random.normal(loc=0.0, scale=sigma)
            xs.append(x)
            ys.append(y)

    xs = np.array(xs).reshape((-1, 1))
    ys = np.array(ys).reshape((-1, 1))
    mus = np.array(mus).reshape((-1, 1))
    ss = np.array(ss).reshape((-1, 1))

    lb = np.array([0.0])
    ub = np.array([10.0])

    variogram = kriging.Variogram(xs, ys, lb, ub)
    variogram.theoretical.fit()

    # -- fitting nugget

    bounds = [(1e-12, 0.99)]
    differential_evolution(func=_get_obj, bounds=bounds)

    variogram.plot()
    print(variogram.theoretical)

    # -- plot

    kparam = kriging.Kparam(variogram)

    xs_new = np.reshape(np.linspace(start=0, stop=10, num=50, endpoint=True), newshape=(-1,1))
    ys_new, vs_new = kriging.predict(xs_new, kparam)
    gs_new = np.sqrt(vs_new)

    fig = plt.figure(figsize=(6.4*2, 4.8))
    ax1 = fig.add_subplot(121)
    ax1.scatter(xs, ys, marker=".", s=10, c="c")

    ax1.plot(xs_new, mus, c="g")  # data mu
    ax1.plot(xs_new, mus+ss, c="r", linestyle='dashed')  # data sigma
    ax1.plot(xs_new, mus-ss, c="r", linestyle='dashed')  # data sigma

    ax1.plot(xs_new, ys_new, c="b")  # kriging my
    ax1.plot(xs_new, ys_new+gs_new, c="y", linestyle='dashed')  # kriging sigma
    ax1.plot(xs_new, ys_new-gs_new, c="y", linestyle='dashed')  # kriging sigma

    ax1.set_title("kriging mean")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")

    ax2 = fig.add_subplot(122)
    ax2.plot(xs_new, vs_new)
    ax2.set_title("kriging variance")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$\sigma^2$")
    plt.show()
