from init import *
import time

def interaction_force(x,y,theta,eta):
    # x-matrices for vectorization X1 = [x,x,x,x,x], X2 = [x.T;x.T; ...]
    X1 = np.reshape(x, (N, 1)) * O1N
    X2 = ON1 * np.reshape(np.reshape(x, (N, 1)), (1, N))
    # y-matrices for vectorization Y1 = [y,y,y,y], Y2 = [y.T;y.T; ...]
    Y1 = np.reshape(y, (N, 1)) * O1N
    Y2 = ON1 * np.reshape(np.reshape(y, (N, 1)), (1, N))
    
    # compute interaction terms
    XX = X1 - X2 # difference in the x-direction
    YY = Y1 - Y2 # difference in the y-direction
    
    # peridicity: consider periodic boundary conditions on unit square, i.e. [0,1]^2
    YY -= (abs(YY)>=(xmax-xmin)/2.0)*(YY>0.0)*(xmax-xmin)
    YY += (abs(YY)>=(xmax-xmin)/2.0)*(YY<=0.0)*(xmax-xmin)
    XX -= (abs(XX)>=(xmax-xmin)/2.0)*(XX>0.0)*(xmax-xmin)
    XX += (abs(XX)>=(xmax-xmin)/2.0)*(XX<=0.0)*(xmax-xmin)

    # square the differences
    XX2 = np.power(XX,2)
    YY2 = np.power(YY,2)

    # compute the interaction terms, r denots the repulsive part, a the attractive part
    fr = np.multiply((alpha * (eta**2)* (XX2 + YY2) + beta), np.exp(-eR * eta*np.sqrt(XX2 + YY2)))
    fa = -gam * eta* np.sqrt(XX2 + YY2) * np.exp(-eA * eta*np.sqrt(XX2 + YY2))
    FR1 = np.multiply(fr, eta*XX)
    FR2 = np.multiply(fr, eta*YY)
    RTRD1 = (np.cos(theta)**2 + chi*np.sin(theta)**2)*eta*XX + (1-chi)*np.sin(theta)*np.cos(theta)*eta*YY
    RTRD2 = (1-chi)*np.sin(theta)*np.cos(theta)*eta*XX + (np.sin(theta)**2 + chi * np.cos(theta)**2)*eta*YY
    FA1 = fa*RTRD1
    FA2 = fa*RTRD2
    F1 = FR1 + FA1
    F2 = FR2 + FA2
    # reshape to make it compatible
    hpx = np.reshape(np.sum(F1, 1), (N, 1))
    hpy = np.reshape(np.sum(F2, 1), (N, 1))
    # return the x- and y-component of the interaction force
    return hpx, hpy

def solve_forward(x, y, dt, T, theta, eta, save=False):
    # write some output to the log file
    f_log = open(path + "log_file.txt", "a")
    f_log.write('forward theta {0} eta {1}\n'.format(theta, eta))
    f_log.close()
    # copy initial condition
    x1 = x.copy()
    y1 = y.copy()
    # store to output vector
    xvec = x1.copy()
    yvec = y1.copy()

    k = 0 # loop counter
    t = 0.0 # time variable
    while(t<T):

        # compute interaction force
        hpx, hpy = interaction_force(x1, y1, theta, eta)

        # add scaled interaction terms
        x1 += 1./N * hpx * dt
        y1 += 1./N * hpy * dt

        # periodicity vectorized
        x1 -= (xmax - xmin)*(x1 > xmax)
        x1 += (xmax - xmin)*(x1 < xmin)
        y1 -= (xmax - xmin)*(y1 > xmax)
        y1 += (xmax - xmin)*(y1 < xmin)

        # store values for later computations
        xvec = np.hstack((xvec, x1))
        yvec = np.hstack((yvec, y1))

        # update time and counter
        t += dt
        k += 1

    # if save flag is active store solution (especially needed for reference solution)
    if save:
        np.savetxt(path+'x_opt.txt',xvec[:,k])
        np.savetxt(path+'y_opt.txt',yvec[:,k])

    # return time dependent x- and y-components of the particles
    return xvec, yvec

