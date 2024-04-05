### EXAMPLE QBM EXECUTION ###
# Shows an example of how the QBM can be executed, and how the learning process can be visualised

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from QBM_Main import QBM, weight_mul, make_op_arrays

#%%% EXAMPLE EXECUTION %%%

# Hamiltonian parameters (EDIT HERE)
n = 4    # no. of qubits
beta = 1 # Inverse Boltzmann temperature

# Make spin operators
_, op_ar_x, op_ar_zz = make_op_arrays(n)

# Uncomment to see what the arrays look like
#print("ar zz:")
#print(op_ar_zz)
#print("ar x:")
#print(op_ar_x)

# Create target Hamiltonian (by picking weights randomly from [-1,1))
h_exact = np.random.uniform(-1, 1, n)
J_exact = np.random.uniform(-1, 1, (n,n))
# Make J symmetric & have diagonal 0
J_exact = (J_exact + J_exact.T)/2
for i in range(n):
    J_exact[i,i] = 0

# Uncomment to see what the target weights look like
#print("J_exact:")
#print(J_exact)
#print("h_exact:")
#print(h_exact)

H_exact = weight_mul(J_exact.reshape(-1), op_ar_zz) + weight_mul(h_exact, op_ar_x) # Target Hamiltonian
# Uncomment to see what the Hamiltonian looks like
print("H_exact:")
print(H_exact)

eta = expm(-beta*H_exact)/expm(-beta*H_exact).trace() # Target density matrix
# Uncomment to see what the target density matrix looks like
#print("eta:")
#print(eta)

# Loss for exact parameters (minimum loss)
opt_loss = beta * (eta @ H_exact).trace() + np.log(expm(-beta * H_exact).trace())

# Generate & learn QBM
MyQBM = QBM(eta, n, 1)
#np.random.seed(15163)   # To fix initialization (useful for comparing different parameters with the same initialization)
MyQBM.learn('GD', noise=0, precision=1e-4, epsilon=0.2)

# Uncomment to see what the learned weights look like & how long learning took
print("Learned J:")
print(MyQBM._J)
print("Learned h:")
print(MyQBM._h)
print("Learning stopped after:")
print(str(MyQBM.qbm_it) + " iterations")

#%%% PLOT %%%

plt.figure(figsize = (16,12))

# Plot parameters (EDIT HERE)
close_up = False    # Whether to show close-ups or the full plot
eps = 0.05          # For close-ups: how far the plot extends from the true value
xlim = MyQBM.qbm_it # To limit x axis on the right side

# Plot model loss - optimal loss
plt.subplot(2,2,1)
plt.plot(MyQBM.loss_QBM_track - opt_loss, label="Model loss - Optimal loss")
plt.yscale("log")
plt.title("Evolution of QBM loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
if(close_up):
    plt.ylim(-eps, eps) # For a close-up
plt.xlim((0,xlim))
plt.legend()

# Plot average absolute gradient
plt.subplot(2,2,2)
plt.plot(MyQBM.grad_QBM_track, label="Average absolute gradient")
plt.yscale("log")
plt.title("Evolution of average absolute gradient")
plt.xlabel("Iteration")
plt.ylabel("Average absolute gradient")
if(close_up):
    plt.ylim(-eps, eps) # For a close-up
plt.xlim((0,xlim))
plt.legend()

# Plot model <sigma_x> vs target <sigma_x>
plt.subplot(2,2,3)
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

# First four model J vs target J
#plt.subplot(2,2,3)
#plt.plot(np.array(QBM.J_QBM_track).reshape(-1), label=["model J[{}]".format(i) for i in range(4)])
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

# Model weights h vs target h
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

plt.show()