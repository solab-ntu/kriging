import matplotlib.pyplot as plt
import numpy as np

import kriging

if __name__ == "__main__":

    def gomez(x: tuple) -> float:
        f = (4-2.1*(x[0]**2)+(x[0]**4)/3)*(x[0]**2) + x[0]*x[1] + (-4+4*(x[1]**2))*(x[1]**2)
        return f

    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    num = 5

    x1 = np.linspace(lb[0], ub[0], num)
    x2 = np.linspace(lb[1], ub[1], num)
    x1g, x2g = np.meshgrid(x1, x2)
    xs = np.hstack((x1g.reshape(-1,1), x2g.reshape(-1,1)))
    ys = np.array([gomez(x) for x in xs]).reshape((-1,1))

    variogram = kriging.Variogram(xs, ys, lb, ub)
    variogram.theoretical.fit()
    variogram.plot()

    kparam = kriging.Kparam(variogram)

    # -- plot

    num = 50
    x1 = np.linspace(lb[0], ub[0], num)
    x2 = np.linspace(lb[1], ub[1], num)
    x1g, x2g = np.meshgrid(x1, x2)

    xs_new = np.hstack((x1g.reshape(-1,1), x2g.reshape(-1,1)))
    ys_new, vs_new = kriging.predict(xs_new, kparam)

    fig = plt.figure(figsize=(6.4*2, 4.8))
    
    ax1 = fig.add_subplot(121)
    ct1 = ax1.contour(x1g, x2g, ys_new.reshape((num, num)), cmap=plt.cm.jet)
    ax1.scatter(xs[:,0], xs[:,1], marker="^", s=20, c="r")
    ax1.set_title("kriging mean")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    fig.colorbar(ct1)

    ax2 = fig.add_subplot(122)
    ct2 = ax2.contour(x1g, x2g, vs_new.reshape((num, num)), cmap=plt.cm.jet)
    ax2.scatter(xs[:,0], xs[:,1], marker="^", s=20, c="r")
    ax2.set_title("kringing variance")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    fig.colorbar(ct2)

    plt.show()
