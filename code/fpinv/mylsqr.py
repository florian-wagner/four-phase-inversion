from math import sqrt
import numpy as np
from numpy.linalg import norm
import pygimli as pg


def lsqr(A, b, damp=0.0, x=None, maxiter=200, show=False):
    """Solve A x = b in a Least-Squares sense using LSQR algorithm.

    After Paige and Saunders (1982)"""
    if x is None:  # no starting vector
        x = pg.RVector(A.cols())
        u = b
    else:
        u = A.mult(x) - b

    beta = norm(u)
    u /= beta
    v = A.transMult(u)
    alfa = norm(v)
    v /= alfa
    Arnorm0 = alfa * 1.0
    Arnorm = Arnorm0 * 1.0
    w = v.copy()
    phiU = beta
    rhoU = alfa
    for i in range(maxiter):
        if show and (i % 10 == 0):
            print(i, Arnorm, Arnorm/Arnorm0)

        u = A.mult(v) - alfa * u
        beta = norm(u)
        if np.isclose(beta, 0.0):
            break
        u /= beta
        v = A.transMult(u) - beta * v
        alfa = norm(v)
        v /= alfa
        rho = sqrt(rhoU**2 + beta**2)
        c = rhoU / rho
        s = beta / rho
        theta = s * alfa
        rhoU = - c * alfa
        phi = c * phiU
        phiU = s * phiU
        x += (phi/rho) * w
        w = v - (theta/rho) * w
        Arnorm = phiU * alfa * abs(c)
        if Arnorm / Arnorm0 < 1e-8:
            if show:
                print(i, Arnorm, Arnorm/Arnorm0)
            break

    return x
