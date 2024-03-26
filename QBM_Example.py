### EXAMPLE QBM EXECUTION ###
# Shows an example of how the QBM can be executed, and how the learning process can be visualised

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from QBM_Main import QBM, weight_mul, make_op_arrays

#%%% EXAMPLE EXECUTION %%%

n = 4
beta = 1

# Make spin operators
_, op_ar_x, op_ar_zz = make_op_arrays(n)

#print("ar zz:")
#print(op_ar_zz)
#print("ar x:")
#print(op_ar_x)

# Target to approximate (for example, any weights in [-1,1] will work):
# H* = z1z2 - x1 + x2
J_exact = np.random.uniform(-1, 1, (n,n))
J_exact = (J_exact + J_exact.T)/2 # To make J symmetric
for i in range(n):
    J_exact[i,i] = 0
h_exact = np.random.uniform(-1, 1, n)

H_exact = weight_mul(J_exact.reshape(-1), op_ar_zz) + weight_mul(h_exact, op_ar_x)
print("H_exact:")
print(H_exact)

eta = expm(-beta*H_exact)/expm(-beta*H_exact).trace()
print("eta: ")
print(eta)

opt_loss = beta * (eta @ H_exact).trace() + np.log(expm(-beta * H_exact).trace()) # Loss for exact parameters (minimum loss)

MyQBM = QBM(eta, n, 1)

#np.random.seed(15163)   # To fix initialization
MyQBM.learn('Nesterov_Book', precision=1e-4, epsilon=0.2)

print("QBM J:")
print(MyQBM._J)
print("QBM h:")
print(MyQBM._h)
print("Learning stopped after:")
print(str(MyQBM.qbm_it) + " iterations")

#print(MyQBM.stat_QBM_track['sigma_x'])

#%%% PLOT %%%

plt.figure(figsize = (16,12))

close_up = False  # Whether to show close-ups or the full plot
eps = 0.05        # For close-ups: how far the plot extends from the true value
xlim = 200        # To limit x axis on the right side

# Loss
plt.subplot(2,2,1)
plt.plot(MyQBM.loss_QBM_track - opt_loss, label="Model loss - Optimal loss")
#plt.axhline(y=opt_loss, color='C0', linestyle='--', label="Optimal loss")
#plt.yscale("log")
plt.title("Evolution of QBM loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
if(close_up):
    plt.ylim(0.45,0.55) # For a close-up
plt.xlim((0,xlim))
plt.legend()

# Loss difference per step
#plt.subplot(2,2,2)
#plt.plot(QBM.dl_QBM_track)
#plt.yscale("log")
#plt.title("QBM loss difference in each step")
#plt.xlabel("Iteration")
#plt.ylabel("dl")
#plt.legend()

# Model stat vs target (sigma_z)
plt.subplot(2,2,2)
plt.plot([dic['sigma_x'] for dic in MyQBM.stat_QBM_track], label=["model <sigma_x>[{}]".format(i) for i in range(MyQBM.n)])
for i in range(MyQBM.n):
    plt.axhline(y=MyQBM._target_stat['sigma_x'][i], color='C{}'.format(i), linestyle='--', label="true <sigma_x[{}]>".format(i))
plt.title("Expectation value of sigma_x")
plt.xlabel("Iteration")
plt.ylabel("<sigma_x>")
if(close_up):
    plt.ylim(MyQBM._target_stat['sigma_x'][0]-eps, MyQBM._target_stat['sigma_x'][0]+eps) # For a close-up of the first qubit
plt.xlim((0,xlim))
plt.legend()

# Model weights J vs target
#plt.subplot(2,2,3)
#plt.plot(np.array(QBM.J_QBM_track).reshape(-1), label=["model J[{}]".format(i) for i in range(QBM.n)])
#plt.axhline(y=J_exact[0,0], color='C0', linestyle='--', label="true J[0,0]")
#plt.axhline(y=J_exact[0,1], color='C1', linestyle='--', label="true J[0,1]")
#plt.axhline(y=J_exact[1,0], color='C2', linestyle='--', label="true J[1,0]")
#plt.axhline(y=J_exact[1,1], color='C3', linestyle='--', label="true J[1,1]")
#plt.title("Evolution of J")
#plt.xlabel("Iteration")
#plt.ylabel("J_ij")
#if(close_up):
#    plt.ylim(J_exact[0,1]-eps, J_exact[0,1]+eps) # For a close-up of the first interaction term
#plt.legend()

# Model weights h vs target
plt.subplot(2,2,4)
plt.plot(MyQBM.h_QBM_track, label=["model h[{}]".format(i) for i in range(MyQBM.n)])
for i in range(MyQBM.n):
    plt.axhline(y=h_exact[i], color='C{}'.format(i), linestyle='--', label="true h[{}]".format(i))
plt.title("Evolution of h")
plt.xlabel("Iteration")
plt.ylabel("h")
if(close_up):
    plt.ylim(h_exact[0]-eps, h_exact[0]+eps) # For a close-up of the first qubit
plt.xlim((0,xlim))
plt.legend()

#plt.subplot(2,2,4)
#plt.plot(np.array(QBM.J_QBM_track).reshape(-1), label=["w_z[{},{}]".format(i,j) for j in range(QBM.n) for i in range(QBM.n)])
#plt.title("Evolution of J")
#plt.xlabel("Iteration")
#plt.ylabel("J")
#plt.legend()

plt.show()