"""
1-d fitting nugget (one sigma)
"""

import matplotlib.pyplot as plt
import numpy as np

import kriging

if __name__ == "__main__":

    # -- normal dist.
    mus = list()
    ss = list()

    # -- actual sample
    xs = list()
    ys = list()

    lb, ub = 0.0, 10.0

    for i in np.linspace(start=lb, stop=ub, num=50, endpoint=True):
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

    mus = np.array(mus).reshape((-1, 1))
    ss = np.array(ss).reshape((-1, 1))

    xs = np.array(xs).reshape((-1, 1))
    ys = np.array(ys).reshape((-1, 1))

    lb = np.array([lb])
    ub = np.array([ub])

    # -- fitting normal kriging
    variogram = kriging.Variogram(xs, ys, lb, ub)
    variogram.theoretical.fit()
    kriging.fit_nugget(variogram=variogram)

    variogram.plot()

    # -- plot comparison
    kparam = kriging.Kparam(variogram)

    xs_new = np.reshape(np.linspace(start=0, stop=10, num=50, endpoint=True), newshape=(-1,1))
    ys_new, vs_new = kriging.predict(xs_new, kparam)
    gs_new = vs_new**0.5

    # -- # actual sample
    fig = plt.figure(figsize=(6.4*2, 4.8))
    ax1 = fig.add_subplot(121)
    ax1.scatter(xs, ys, marker=".", s=10, c="c")
    ax1.plot(xs_new, mus, c="g")  # data mu
    ax1.plot(xs_new, mus+ss, c="r", linestyle='dashed')  # one sigma upper
    ax1.plot(xs_new, mus-ss, c="r", linestyle='dashed')  # one sigma lower

    # -- # kriging result
    ax1.plot(xs_new, ys_new, c="b")  # kriging mu
    ax1.plot(xs_new, ys_new+gs_new, c="y", linestyle='dashed')  # one sigma upper
    ax1.plot(xs_new, ys_new-gs_new, c="y", linestyle='dashed')  # one sigma lower

    ax1.set_title("kriging mean")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")

    # -- # kriging variance
    ax2 = fig.add_subplot(122)
    ax2.plot(xs_new, vs_new)
    ax2.set_title("kriging variance")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$\sigma^2$")

    plt.show()
