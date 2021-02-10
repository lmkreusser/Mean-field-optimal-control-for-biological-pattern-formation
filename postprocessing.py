import numpy as np
from matplotlib import pyplot as plt
from init import *

# print some output the characterises the current setting
print('theta_for', fak_theta_for, 'theta_ref', fak_theta_ref, 'theta', fak_theta )
print('eta_for', eta_for, 'eta_ref', eta_ref, 'eta', eta )

# load cost data
Jvec  = np.loadtxt(path + 'Jvec.txt')
J1vec  = np.loadtxt(path + 'J1vec.txt')
J2vec  = np.loadtxt(path + 'J2vec.txt')
J3vec  = np.loadtxt(path + 'J3vec.txt')
# load artificial data
x_opt = np.loadtxt(path + 'x_opt.txt')
y_opt = np.loadtxt(path + 'y_opt.txt')
# load initial condition of the parameter identification
x_start = np.loadtxt(path + 'x_start.txt')
y_start = np.loadtxt(path + 'y_start.txt')
# load terminal state of the parameter identification
x_end = np.loadtxt(path + 'x_end.txt')
y_end = np.loadtxt(path + 'y_end.txt')

# plot & save the evolution of the cost functional terms
fig, axs = plt.subplots(2, 2, sharex=True)
axs[0, 0].plot(Jvec)
axs[0, 0].set_title(r'$\mathcal{J}$')
axs[0, 1].plot(J1vec, 'tab:orange')
axs[0, 1].set_title(r'$\mathcal{J}_1$')
axs[1, 0].plot(J2vec, 'tab:green')
axs[1, 0].set_title(r'$\mathcal{J}_2$')
axs[1, 0].set_xlabel('iteration')
axs[1, 0].ticklabel_format(style='sci',scilimits=(0,0))
axs[1, 1].plot(J3vec, 'tab:red')
axs[1, 1].set_title(r'$\mathcal{J}_3$')
axs[1, 1].set_xlabel('iteration')
axs[1, 1].ticklabel_format(style='sci',scilimits=(0,0))
plt.savefig('cost{0}.png'.format(int(sys.argv[1])),bbox_inches='tight')

# plot & save the artificial data and the terminal state of the parameter identification
plt.figure(2)
plt.plot(x_opt,y_opt,'ok',fillstyle = 'none')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_end,y_end,'or',fillstyle = 'none')
plt.savefig('optEnd{0}.png'.format(int(sys.argv[1])))

# plot & save the initial conditions for the parameter identification
plt.figure(3)
fig, ax = plt.subplots()
ax.plot(x_start,y_start,'ob',fillstyle = 'none')
ax.set_title('Start',fontsize=16)
ax.set_xlim((-0.05,1.05))
ax.set_ylim((-0.05,1.05))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
ax.set_xticklabels([-0.2,0.0,0.2,0.4,0.6,0.8,1],fontsize=16)
ax.set_yticklabels([-0.2,0.0,0.2,0.4,0.6,0.8,1],fontsize=16)
plt.savefig('startrev{0}.png'.format(int(sys.argv[1])),bbox_inches='tight')

# plot & save the artifical data and the terminal state of the parameter identification
plt.figure(4)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Desired')
ax2.set_title('Optimal')
ax1.plot(x_opt,y_opt,'ob',fillstyle = 'none')
ax2.plot(x_end,y_end,'or',fillstyle = 'none')
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_ylim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
plt.savefig('patternsrev{0}.png'.format(int(sys.argv[1])),bbox_inches='tight')
