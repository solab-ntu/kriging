import matplotlib.pyplot as plt
import numpy as np

class Likelihood:

    """
    Calculate R (the correlation matrix) & Rinv (its inverse)
    then use them to find beta & sigma2 as a function of theta.
    Finally, compute the likelihood function to be maximized by the
    main program, fitkrige.m.  Note that the optimization searches for
    ln(theta) instead of searching for theta directly and for 1x(10^nugget)
    instead of searching for nugget directly.
    """

    def __init__(self):
        pass