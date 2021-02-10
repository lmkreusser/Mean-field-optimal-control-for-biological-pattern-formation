from init import *

def compute_gradient(sx, sy, x, y, theta, eta):


    grad1 = gamma2 * (theta - theta_ref)  # theta part
    grad2 = gamma3 * (eta - eta_ref)      # eta part

    k = 0
    t = 0.0
    while t < T:

        # x-matrices for vectorization X1 = [x,x,x,x,x], X2 = [x.T;x.T; ...]
        X1 = np.reshape(x[:,k], (N, 1)) * O1N
        X2 = ON1 * np.reshape(np.reshape(x[:,k], (N, 1)), (1, N))
        # y-matrices for vectorization Y1 = [y,y,y,y], Y2 = [y.T;y.T; ...]
        Y1 = np.reshape(y[:,k], (N, 1)) * O1N
        Y2 = ON1 * np.reshape(np.reshape(y[:,k], (N, 1)), (1, N))
        # compute interaction terms
        XX = X1 - X2
        YY = Y1 - Y2
        
        
        # periodicity: consider periodic boundary conditions on unit square, i.e. [0,1]^2
        YY -= (abs(YY)>=(xmax-xmin)/2.0)*(YY>0.0)*(xmax-xmin)
        YY += (abs(YY)>=(xmax-xmin)/2.0)*(YY<=0.0)*(xmax-xmin)
        XX -= (abs(XX)>=(xmax-xmin)/2.0)*(XX>0.0)*(xmax-xmin)
        XX += (abs(XX)>=(xmax-xmin)/2.0)*(XX<=0.0)*(xmax-xmin)

        # square the differences
        XX2 = np.power(XX, 2)
        YY2 = np.power(YY, 2)

        # terms w.r.t eta
        # coefficient functions fr_1,fa_1 of forces
        fr_1 = np.multiply((alpha * (eta**2) * (XX2 + YY2) + beta), np.exp(-eR * eta * np.sqrt(XX2 + YY2)))
        fa_1 = -gam * eta*np.sqrt(XX2 + YY2) * np.exp(-eA * eta * np.sqrt(XX2 + YY2))
        FR1_1 = np.multiply(fr_1, XX)
        FR2_1 = np.multiply(fr_1, YY)
        RTRD1_1 = (np.cos(theta) ** 2 + chi * np.sin(theta) ** 2) * XX + (1 - chi) * np.sin(theta) * np.cos(theta) * YY
        RTRD2_1 = (1 - chi) * np.sin(theta) * np.cos(theta) * XX + (np.sin(theta) ** 2 + chi * np.cos(theta) ** 2) * YY
        FA1_1 = np.multiply(fa_1, RTRD1_1)
        FA2_1 = np.multiply(fa_1, RTRD2_1)
        F1_1 = FR1_1 + FA1_1 #note that the factor eta vanishes now
        F2_1 = FR2_1 + FA2_1
        # second part due to chain rule
        fr = np.multiply(2*alpha * eta * (XX2 + YY2)-eR*np.sqrt(XX2 + YY2)*(alpha * (eta**2)*(XX2 + YY2) + beta), np.exp(-eR * eta * np.sqrt(XX2 + YY2)))
        fa = (-gam*np.sqrt(XX2 + YY2) +(-eA*np.sqrt(XX2 + YY2))*(-gam) * eta*np.sqrt(XX2 + YY2)) * np.exp(-eA * eta * np.sqrt(XX2 + YY2))
        FR1 = np.multiply(fr, eta*XX)
        FR2 = np.multiply(fr, eta*YY)
        RTRD1 = (np.cos(theta) ** 2 + chi * np.sin(theta) ** 2) *eta* XX + (1 - chi) * np.sin(theta) * np.cos(theta) * eta * YY
        RTRD2 = (1 - chi) * np.sin(theta) * np.cos(theta) * eta* XX + (np.sin(theta) ** 2 + chi * np.cos(theta) ** 2) * eta* YY
        FA1 = fa * RTRD1
        FA2 = fa * RTRD2
        F1 = FR1 + FA1 + F1_1
        F2 = FR2 + FA2 + F2_1

        grad2 += -1./N*np.sum(np.reshape(np.sum(F1, 1), (N, 1))*np.reshape(sx[:, k], (N, 1)) \
                              + np.reshape(np.sum(F2, 1), (N, 1))*np.reshape(sy[:, k], (N, 1)))*dt

        # terms w.r.t. theta
        M11 = (chi-1.) * np.cos(theta)*np.sin(theta) * XX - (np.sin(theta)**2 + chi * np.cos(theta)**2) * YY
        M12 = (np.cos(theta)**2 + chi*np.sin(theta)**2) * XX + (1.-chi)*np.cos(theta)*np.sin(theta) * YY

        M21 = (chi -1.) * np.cos(theta) * np.sin(theta) * XX + (np.cos(theta)**2 + chi * np.sin(theta)**2) * YY
        M22 = -(np.sin(theta)**2 + chi * np.cos(theta)**2) * XX + (1. - chi) * np.sin(theta) * np.cos(theta) * YY

        G1 = eta*np.reshape(np.sum(np.multiply(fa_1, M11 + M21), 1), (N, 1))
        G2 = eta*np.reshape(np.sum(np.multiply(fa_1, M12 + M22), 1), (N, 1))

        grad1 += -1./N*np.sum(G1*np.reshape(sx[:, k], (N, 1)) + G2*np.reshape(sy[:, k], (N, 1)))*dt

        # update time and counter
        k += 1
        t += dt

    # return gradient components for theta and eta
    return grad1, grad2
