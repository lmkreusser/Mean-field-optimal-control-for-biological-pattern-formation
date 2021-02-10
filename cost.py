from init import *


def evaluate_costfunctional(x,y,theta,eta,x0,y0):

    a, b = np.ones((N,)) / N, np.ones((N,)) / N  # uniform distribution on samples

    xs = np.hstack((x0, y0))
    xt = np.hstack((np.reshape(x[:, -1], (N, 1)), np.reshape(y[:, -1], (N, 1))))

    # loss matrix
    M = ot.dist(xt, xs)  # W_2 metric with the help of OT library

    # %% EMD
    J1 = 0.5 * N * gamma1  * ot.emd2(a, b, M)  # emd2 is W_2 squared
    J2 = 0.5 * gamma2 * (theta - theta_ref)**2 # first control term
    J3 = 0.5 * gamma3 * (eta - eta_ref)**2     # second control term

    return J1, J2, J3, J1 + J2 + J3
