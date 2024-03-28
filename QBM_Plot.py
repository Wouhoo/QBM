### PLOT RESULTS OF TESTS ###
# Various ways of plotting the results of the QBM tests

import numpy as np
import matplotlib.pyplot as plt
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web

# Directory where the data may be found
filedir = "./Data"

#%%% PLOT RESULTS OF TESTS (INDIVIDUAL HAMILTONIANS) %%%
'''
f = h5py.File(filedir + '\\Data.hdf5','r')

plt.figure(figsize = (16,8))

close_up = True  # Whether to show close-ups or the full plot
eps = 0.05       # For close-ups: how far the plot extends from the true value

# Parameters for which to compare the optimizers
model = "Random Ising model"
n = 4
H_number = 3
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR'] # Which optimizers to compare

# Retrieve data
H_group = f["{}/n = {}/Hamiltonian {}".format(model, n, H_number)]
print("Hamiltonian parameters:")
print(H_group.attrs['H_params_w'])
print(H_group.attrs['H_params_b'])

iters_per_opt = []

# Plot losses while also filling up the array with total iterations (for a bar plot later)
plt.subplot(1,2,1)
final_loss = 0
for optimizer in optimizer_list:
    opt_group= H_group[optimizer]
    
    loss = opt_group["Loss"][()]
    if(close_up and optimizer == 'Nesterov_GR'):
        final_loss = loss[-1] # For easier plotting of close-up: save the minimum loss value which Nesterov_GR (and all other optimizers) converge(s) to
    plt.plot(loss, label=optimizer)
    
    iters = opt_group["Total iterations"][()]
    iters_per_opt.append(iters)

plt.title("Evolution of QBM loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
if(close_up):
    plt.ylim((final_loss-0.01,final_loss+0.05))
    plt.xlim((0,100))

plt.subplot(1,2,2)
plt.bar(optimizer_list, iters_per_opt, color=['C{}'.format(i) for i in range(len(optimizer_list))])
plt.title("Number of iterations to reach desired precision")
plt.xlabel("Optimizer")
plt.ylabel("Iterations")

f.close()
plt.show()
'''    

#%%% PLOT RESULTS OF TESTS (ITERATIONS VS PRECISION) %%%

# Parameters for which to compare the optimizers
prec_list = ["1e-2", "1e-3", "1e-4", "1e-5"]
model = "Random Ising model"
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']

n_list = [2,4,6,8]
H_count = 5

# Data files
f = h5py.File(filedir + '\\Data_random_until1e-5.hdf5', 'r')

# Plot nr of iterations as a function of precision for each optimizer


for optimizer in optimizer_list:
    total_iters = []
    
    for precision in prec_list:
        iters = []
        
        for n in n_list:
            for H_counter in range(1,H_count+1):
                path = '{}/precision = {}/n = {}/Hamiltonian {}/{}'.format(model, precision, n, H_counter, optimizer)
                iters.append(f[path + '/Total iterations'][()])
        
        total_iters.append(np.average(iters)) # No. of iterations for this (precision, optimizer) combo averaged over all n and H
        
    plt.plot(prec_list, total_iters, marker='o', label=optimizer) # Add line for this optimizer to plot

plt.xlabel("Precision")
plt.ylabel("No. of iterations")
plt.legend()      
plt.show()