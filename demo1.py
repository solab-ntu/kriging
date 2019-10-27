import matplotlib.pyplot as plt
import numpy as np

import kriging

if __name__ == "__main__":

    xs = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    ys = np.array([0, 1, 2, 1, 4, 7, 6]).reshape((-1, 1))

    lb = np.array([0])
    ub = np.array([6])

    variogram = kriging.Variogram(xs, ys, lb, ub)
    variogram.theoretical.fit()
    variogram.plot()

    kparam = kriging.Kparam(variogram)

    # -- plot

    xs_new = np.reshape(np.linspace(start=0, stop=6, num=50, endpoint=True), newshape=(-1,1))
    ys_new, vs_new = kriging.predict(xs_new, kparam)
    
    fig = plt.figure(figsize=(6.4*2, 4.8))
    ax1 = fig.add_subplot(121)
    ax1.plot(xs_new, ys_new)
    ax1.scatter(xs, ys, marker="^", s=20, c="r")
    ax1.set_title("kriging mean")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax2 = fig.add_subplot(122)
    ax2.plot(xs_new, vs_new)
    ax2.set_title("kriging variance")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$\sigma^2$")
    plt.show()
