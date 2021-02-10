from init import *


def plot_stuff(G0,xt,xs,x0,rhx,y0,rhy):
    # plots for debugging purposes
    pl.figure(3)
    pl.imshow(G0, interpolation='nearest')
    pl.title('OT matrix G0')

    pl.figure(4)
    ot.plot.plot2D_samples_mat(xt, xs, G0, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.plot(np.reshape(xt[:, 0], (N, 1))+rhx, np.reshape(xt[:, 1], (N, 1))+ rhy, '+g')
    pl.legend(loc=0)
    pl.title('OT matrix with samples')

    pl.show()

def adjoint_interaction_force(x, y, sx, sy, theta, eta, G0 ):

    # x-matrices for vectorization X1 = [x,x,x,x,x], X2 = [x.T;x.T; ...]
    X1 = np.reshape(x, (N, 1)) * O1N
    X2 = ON1 * np.reshape(np.reshape(x, (N, 1)), (1, N))
    # y-matrices for vectorization Y1 = [y,y,y,y], Y2 = [y.T;y.T; ...]
    Y1 = np.reshape(y, (N, 1)) * O1N
    Y2 = ON1 * np.reshape(np.reshape(y, (N, 1)), (1, N))
    # compute interaction terms
    XX = X1 - X2
    YY = Y1 - Y2
    
    # periodicity: consider periodic boundary conditions on unit square, i.e. [xmin,xmax]^2
    YY -= (abs(YY)>=(xmax-xmin)/2.0)*(YY>0.0)*(xmax-xmin)
    YY += (abs(YY)>=(xmax-xmin)/2.0)*(YY<=0.0)*(xmax-xmin)
    XX -= (abs(XX)>=(xmax-xmin)/2.0)*(XX>0.0)*(xmax-xmin)
    XX += (abs(XX)>=(xmax-xmin)/2.0)*(XX<=0.0)*(xmax-xmin)

    # compute square of the differences
    XX2 = XX ** 2
    YY2 = YY ** 2

    # interaction terms
    R = np.sqrt(XX2 + YY2)
    R1 = R +  np.identity(N)
    dxfr_fac = 2.*alpha*(eta**2)*np.exp(-eR * eta * R) + (alpha*(eta**2)*(XX2 + YY2) + beta) * np.multiply(-np.divide(eR*eta, R1), np.exp(-eR*eta*R))
    dxfr1 = np.multiply(dxfr_fac, XX)
    dxfr2 = np.multiply(dxfr_fac, YY)

    dxfa_fac = (-gam*eta*np.divide(np.exp(-eA*eta*R), R1) + gam*eA*(eta**2)*np.exp(-eA*eta*R))
    dxfa1 = dxfa_fac*XX
    dxfa2 = dxfa_fac*YY

    RTRD1 = (np.cos(theta) ** 2 + chi * np.sin(theta) ** 2) * eta*XX + (1 - chi) * np.sin(theta) * np.cos(theta) * eta*YY
    RTRD2 = (1 - chi) * np.sin(theta) * np.cos(theta) * eta*XX + (np.sin(theta) ** 2 + chi * np.cos(theta) ** 2) * eta*YY

    # first part of the adjoint
    A11 = dxfa1*RTRD1
    A21 = dxfa1*RTRD2
    A12 = dxfa2*RTRD1
    A22 = dxfa2*RTRD2

    fa = -gam * eta*np.sqrt(XX2 + YY2) * np.exp(-eA * eta*np.sqrt(XX2 + YY2))
    fr = np.multiply((alpha*(eta**2)*(XX2 + YY2) + beta), np.exp(-eR*eta*np.sqrt(XX2 + YY2)))

    # second part of the adjoint
    B11 = fa * (np.cos(theta) ** 2 + chi * np.sin(theta) ** 2)*eta
    B12 = fa * (1 - chi) * np.sin(theta) * np.cos(theta)*eta
    B21 = fa * (1 - chi) * np.sin(theta) * np.cos(theta)*eta
    B22 = fa * (np.sin(theta) ** 2 + chi * np.cos(theta) ** 2)*eta

    # third part of the adjoint
    C11 = np.multiply(dxfr1, XX)
    C21 = np.multiply(dxfr1, YY)
    C12 = np.multiply(dxfr2, XX)
    C22 = np.multiply(dxfr2, YY)

    # put the system matrix together
    M11 = eta*(A11 + B11 + C11 + fr)
    M12 = eta*(A12 + B12 + C12)
    M21 = eta*(A21 + B21 + C21)
    M22 = eta*(A22 + B22 + C22 + fr)

    # return system matrix of implicit scheme
    return M11, M12, M21, M22

def solve_adjoint(x, y, dt, T, theta, eta, x0, y0):

    # compute OT from current configuration to desired configuration
    a, b = np.ones((N,)) / N, np.ones((N,)) / N  # uniform distribution on samples

    # set up the two samples to compute the optimal transport plan
    xs = np.hstack((x0, y0))
    xt = np.hstack((np.reshape(x[:, -1], (N, 1)), np.reshape(y[:, -1], (N, 1))))

    # extract vectors for right-hand side of adjoint
    # x-matrices for vectorization X1 = [x,x,x,x,x], X2 = [x.T;x.T; ...]
    X1 = np.reshape(x[:, -1], (N, 1)) * O1N
    X2 = ON1 * np.reshape(np.reshape(x0, (N, 1)), (1, N))

    # y-matrices for vectorization Y1 = [y,y,y,y], Y2 = [y.T;y.T; ...]
    Y1 = np.reshape(y[:, -1], (N, 1)) * O1N
    Y2 = ON1 * np.reshape(np.reshape(y0, (N, 1)), (1, N))
    # compute rhs from optimal tranport plan
    XX = (X2-X1)
    YY = (Y2-Y1)
    
    # periodicity: consider periodic boundary conditions on unit square, i.e. [xmin,xmax]^2
    
    YY -= (abs(YY)>=(xmax-xmin)/2.0)*(YY>0.0)*(xmax-xmin)
    YY += (abs(YY)>=(xmax-xmin)/2.0)*(YY<=0.0)*(xmax-xmin)
    XX -= (abs(XX)>=(xmax-xmin)/2.0)*(XX>0.0)*(xmax-xmin)
    XX += (abs(XX)>=(xmax-xmin)/2.0)*(XX<=0.0)*(xmax-xmin)
    M = XX**2+YY**2

    M /= M.max() # rescale for computation

    #%% EMD
    G0 = ot.emd(a, b, M) # G0 contains the matches

    # right-hand side of adjoints
    rhx = np.reshape(np.sum(XX*(G0>0.), axis=1), (N, 1))
    rhy = np.reshape(np.sum(YY*(G0>0.), axis=1), (N, 1))

    # initialze the adjoints using rhs
    sx = gamma1*rhx/N # multiply by /N due to rescaling of adjoints
    sy = gamma1*rhy/N # multiply by /N due to rescaling of adjoints

    # initialze for data storage
    sxvec = sx
    syvec = sy

    k = 0 # loop counter
    t = 0.0 # time variable
    while (t < T):
        # compute the system matrix for the implicit scheme
        M11, M12, M21, M22 = adjoint_interaction_force(x[:, k], y[:, k], sx, sy, theta, eta, G0)

        #full-implizit solver
        MZ = 0.0 * np.identity(2 * N)
        MZ[:N, :N] = dt * 1. / N * (np.diag(np.sum(M11, axis=1)) - M11)
        MZ[:N, N:] = dt * 1. / N * (np.diag(np.sum(M21, axis=1)) - M21)
        MZ[N:, N:] = dt * 1. / N * (np.diag(np.sum(M22, axis=1)) - M22)
        MZ[:N, N:] = dt * 1. / N * (np.diag(np.sum(M12, axis=1)) - M12)

        B = np.identity(2*N) + MZ

        srhsv = np.ones(2*N)
        srhsv[:N] = sx[:,0]
        srhsv[N:] = sy[:,0]
        # solve implicit scheme
        news = np.linalg.solve(B,srhsv)

        sx[:,0] = news[:N]
        sy[:,0] = news[N:]

        # store data for gradient computations
        # rescaling mean-field + linear system
        sx = sx
        sy = sy
        # invert time!
        sxvec = np.hstack((sx, sxvec))
        syvec = np.hstack((sy, syvec))
        # update time and counter
        t += dt
        k += 1
    sxvec = sxvec * N
    syvec = syvec * N
    # write info to log file
    f_log = open(path + "log_file.txt", "a")
    f_log.write('adjoint simulation done! \n')
    f_log.close()
    # return x- and y-components of adjoint over time
    return sxvec, syvec
