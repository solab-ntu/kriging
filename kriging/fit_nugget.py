import numpy as np
from scipy.optimize import differential_evolution
from .kparam import Kparam
from .predict import predict

def _get_nugget_obj(x: tuple, xs: np.array, ys: np.array, total: int, variogram: "kriging.Variogram") -> float:
    criterion = 0.68
    variogram.theoretical.nugget = x[0]
    kparam = Kparam(variogram=variogram)
    ys_new, vs_new = predict(xs, kparam)
    gs_new = vs_new**0.5
    count = 0
    for y, yn, gn in zip(ys, ys_new, gs_new):
        if yn-gn < y < yn+gn:
            count += 1
    obj =  1.0
    if float(count) / float(total) > criterion:
        obj = x[0]
    return obj

def fit_nugget(variogram: "kriging.Variogram"):
    xs = variogram.xs
    ys = variogram.ys
    total = xs.shape[0]
    bounds = [(1e-12, 0.99)]
    res = differential_evolution(func=_get_nugget_obj, bounds=bounds, args=(xs, ys, total, variogram))
    variogram.theoretical.nugget = res.x[0]
