from init import *
from forward import *
from adjoint_implicit import *
from gradient import *
from cost import *
import time


def FDAD(x,y,x0,y0):
    # this function computes the gradient of the cost via finite difference and the gradient of the cost via adjoint
    # you can check the code, we recommend to use a small time interval to speed up the computation

    # change control parameter a little bit
    eta = eta_ref + .1
    theta = theta_ref + 0.1

    # evaluate cost for new control parameter
    x1vec, y1vec = solve_forward(x, y, dt, T, theta, eta)
    J1, J2, J3, Ja = evaluate_costfunctional(x1vec, y1vec, theta, eta, x0, y0)

    # alternate direction and evaluate cost again
    epsilon = 0.0001
    h = 1.
    xbvec, ybvec = solve_forward(x, y, dt, T, theta + epsilon*h, eta + epsilon*h)
    J1, J2, J3, Jb = evaluate_costfunctional(xbvec, ybvec, theta + epsilon*h, eta + epsilon*h, x0, y0)

    # compute finite difference for these two evaluations
    FD = (Jb-Ja)/epsilon

    # compute adjoint for first chi
    sx, sy = solve_adjoint(x1vec, y1vec, dt, T, theta, eta, x0, y0)
    
    # compute grad based on first state and its adjoint
    grad1, grad2 = compute_gradient(sx, sy, x1vec, y1vec, theta, eta)
    # compute gradient*alternating direction
    AD = grad1*h + grad2*h
    # output for comparison
    print('FD', FD, 'AD', AD, 'diff', FD-AD)
    print('FDAD done')


def line_search(x1,y1,theta,eta,grad1,grad2,x0,y0):
    # use simple line search to find correct step size
    xs, ys = np.reshape(x1[:, 0], (N, 1)), np.reshape(y1[:, 0], (N, 1))

    # initalize step size as 1
    step_size = 1.
    # reference cost-functional value
    J1, J2, J3, Jref = evaluate_costfunctional(x1, y1, theta, eta,x0,y0)

    # gradient step
    ntheta = theta - step_size*grad1
    # without projection
    # neta = eta - step_size * grad2
    # with projection
    neta = min(1.1, max(0.9, eta - step_size*grad2))


    # evaluate with new control
    x, y = solve_forward(xs, ys, dt, T, ntheta, neta)
    J1, J2, J3, nJ = evaluate_costfunctional(x, y, ntheta, neta,x0,y0)

    # if update not successful do a line search
    while Jref < nJ:
        step_size *= 0.5 # make step size smaller
        # compute new update with smaller stepsize
        ntheta = theta - step_size * grad1
        # without projection
        # neta = eta - step_size * grad2
        # with projection
        neta = min(1.1, max(0.9, eta - step_size * grad2))

        # evaluate cost with new control
        x, y = solve_forward(xs, ys, dt, T, ntheta, neta)
        # compute cost
        J1, J2, J3, nJ = evaluate_costfunctional(x, y, ntheta, neta,x0,y0)


    return ntheta, neta, J1, J2, J3, nJ, x, y

# helper function to update the vectors with cost functional values
def Jappend(Jvec,J1vec,J2vec,J3vec,J,J1,J2,J3):
    Jvec.append(J)
    J1vec.append(J1)
    J2vec.append(J2)
    J3vec.append(J3)
    return Jvec,J1vec,J2vec,J3vec


def optimize(x1,y1,theta,eta,x0,y0):

    # this optimizes the problem at hand

    # initialize vector for cost functional values (to be plotted at the end)
    Jvec = []
    J1vec = []
    J2vec = []
    J3vec = []

    # evaluate cost functional
    x, y = solve_forward(x1, y1, dt, T, theta, eta)
    np.savetxt(path+'x_start.txt', x[:, -1])
    np.savetxt(path+'y_start.txt', y[:, -1])
    J1, J2, J3, J = evaluate_costfunctional(x, y, theta, eta,x0,y0)


    # store initial cost value
    Jvec,J1vec,J2vec,J3vec = Jappend(Jvec,J1vec,J2vec,J3vec,J,J1,J2,J3)
    f_log = open(path + "log_file.txt", "a")
    f_log.write('Start J = {0} \n'.format(J))
    f_log.write('J1 {0} J2 {1} J3 {2}\n'.format(J1,J2,J3))
    f_log.close()

    # compute adjoint and gradient
    sx, sy = solve_adjoint(x, y, dt, T, theta, eta, x0, y0)
    grad1, grad2 = compute_gradient(sx, sy, x, y, theta, eta)
    f_log = open(path + "log_file.txt", "a")
    f_log.write('grad1 {0} grad2 {1}\n'.format(grad1,grad2))
    f_log.close()
    # grad stopping criterion
    grad_ref = abs(grad1) + abs(grad2)

    # cost stopping criterion
    Jold = 2.*np.copy(J)

    # relative stopping criterion
    while (abs(grad1) + abs(grad2))/grad_ref > stop:
        # do line search to update the control, get new corresponding values
        theta, eta, J1, J2, J3, J, x, y = line_search(x, y, theta, eta, grad1, grad2, x0, y0)
        np.savetxt(path + 'x_current.txt', x[:, -1])
        np.savetxt(path + 'y_current.txt', y[:, -1])
        f_log = open(path + "log_file.txt", "a")
        f_log.write('new theta {0} eta {1} \n'.format(theta,eta))
        f_log.close()
        # store cost value for post-processing
        Jvec,J1vec,J2vec,J3vec = Jappend(Jvec,J1vec,J2vec,J3vec,J,J1,J2,J3)
        f_log = open(path + "log_file.txt", "a")
        f_log.write('J = {0}\n'.format(J))
        f_log.close()
        # use new values to compute adjoint and gradient
        sx, sy = solve_adjoint(x, y, dt, T, theta, eta, x0, y0)
        grad1, grad2 = compute_gradient(sx, sy, x, y, theta, eta)
        f_log = open(path + "log_file.txt", "a")
        f_log.write('grad1 {0} grad2 {1}\n'.format(grad1,grad2))
        # print stopp value to log file
        stopp_val = (abs(grad1) + abs(grad2)) / grad_ref
        f_log.write('stopp {0}\n'.format(stopp_val))
        f_log.close()



    # save final commands to the log file
    f_log = open(path + "log_file.txt", "a")
    f_log.write('Optimization done!\n')
    f_log.write('theta = {0} eta = {1}\n'.format(theta, eta))
    f_log.write('Jvec {0}\n'.format(Jvec))
    f_log.close()
    # save terminal configuration of the particles
    np.savetxt(path+'x_end.txt', x[:, -1])
    np.savetxt(path+'y_end.txt', y[:, -1])
    # save cost vector for post-processing
    np.savetxt(path+'Jvec.txt',Jvec)  # sum of all cost functional terms
    np.savetxt(path + 'J1vec.txt', J1vec) # first term of cost functional
    np.savetxt(path + 'J2vec.txt', J2vec) # second term of cost functional
    np.savetxt(path + 'J3vec.txt', J3vec) # third term of cost functional

def compute_ref_solution(x1,y1):
    xr, yr = solve_forward(x1, y1, dt, T, theta_for, eta_for, True)
    #xr, yr = solve_forward(x1, y1, dt, T, 0.25*np.pi, 3., True)
    # save opt values


# generate artificial data for paramter identification
compute_ref_solution(x,y)


# set desired values for parameter identification
x0 = np.reshape(np.loadtxt(path+'x_opt.txt'), (N, 1))  # simulated with chi = 0.1, T = 0.01
y0 = np.reshape(np.loadtxt(path+'y_opt.txt'), (N, 1))

# compute solution of the parameter identification
optimize(x,y,theta,eta,x0,y0)
