### PLOT RESULTS OF TESTS ###
# Various ways of plotting the results of the QBM tests. 
# Uncomment whichever plot you want (may have to split this into different files or find some way to make this more convenient later)

import numpy as np
import matplotlib.pyplot as plt
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from QBM_Main import filedir

#%%% PLOT RESULTS OF TESTS (INDIVIDUAL HAMILTONIANS) %%%
# Shows loss evolution and total number of iterations for various optimizers for a specific (precision, n, Hamiltonian) combo.
# This code can also be used as a base for writing code to compare loss and total iterations when varying *one* parameter 
# (for example, comparing the performance of one optimizer for various Hamiltonians of the same model, precision and n)
'''
# Data file (EDIT HERE)
f = h5py.File(filedir + '/Data_random_until1e-6.hdf5','r')

# Parameters for which to compare the optimizers (EDIT HERE)
model = "Random Ising model"                                                           # Currently only the Random Ising model has data
precision = "1e-6"                                                                     # Must be in ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]
n = 6                                                                                  # Must be in [2,4,6,8]
H_number = 2                                                                           # Must be in [1,2,3,4,5] for the Random Ising model
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR'] # Which optimizers to compare

# Plot parameters (EDIT HERE)
close_up = False  # Whether to show close-ups or the full plot
eps = 0.05       # For close-ups: how far the plot extends from the true value
xlim = 100       # For close-ups: where the x-axis (no. of iterations) is cut off

# Retrieve data
H_group = f["{}/precision = {}/n = {}/Hamiltonian {}".format(model, precision, n, H_number)]
# Print Hamiltonian parameters
# These may not be present in the data file depending on how the data was merged
#print("Hamiltonian parameters:")
#print(H_group.attrs['H_params_w'])  
#print(H_group.attrs['H_params_b'])

iters_per_opt = []

plt.figure(figsize = (16,8))

# Plot losses while also filling up the array with total iterations (for a bar plot later)
plt.subplot(1,2,1)
final_loss = 0
for optimizer in optimizer_list:
    opt_group= H_group[optimizer]
    
    loss = opt_group["Model loss - Optimal loss"][()]
    plt.plot(loss, label=optimizer)
    
    iters = opt_group["Total iterations"][()]
    iters_per_opt.append(iters)

# Plot loss evolutions
plt.title("Evolution of QBM loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
if(close_up):
    plt.ylim((-0.01,eps))
    plt.xlim((0,100))

# Plot total iterations for different optimizers
plt.subplot(1,2,2)
plt.bar(optimizer_list, iters_per_opt, color=['C{}'.format(i) for i in range(len(optimizer_list))])
plt.title("Number of iterations to reach desired precision")
plt.xlabel("Optimizer")
plt.ylabel("Iterations")

f.close()
plt.show()
'''    

#%%% PLOT RESULTS OF TESTS (ITERATIONS VS PRECISION) %%%
# Plots *average* number of iterations required by various optimizers for various precisions.
# This code can also be used as a base for writing code to compare total iterations when varying *two or more* parameters
# (for example, comparing the performance of different optimizers for different n, averaging over all models, precisions and Hamiltonians)
#'''
# Parameters for which to compare the optimizers (EDIT HERE)
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR'] # Optimizers to compare
prec_list = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]                                 # Precisions for which to compare the optimizers
model = "Random Ising model"                                                      # Model to do the comparison for
n = 2                                                                             # Amount of qubits to do the comparison for
H_number = 1                                                                           # Hamiltonian to do the comparison for (this corresponds to the J/h ratio in case of the Uniform Ising model)
reverse_plot = False                                                                    # Whether to reverse the x-axis or not

# Data file
f = h5py.File(filedir + '/Data_iters_vs_precision_random.hdf5', 'r')

# Plot nr of iterations as a function of precision for each optimizer
for optimizer in optimizer_list:
    total_iters = f['{}/n = {}/Hamiltonian {}/{}/Total iterations to reach prec_list precisions'.format(model, n, H_number, optimizer)][()]
    plt.plot(np.log10(prec_list), total_iters, marker='o', label=optimizer) # Add line for this optimizer to plot

plt.xlabel("log$_{10}$(Precision)")
plt.ylabel("No. of iterations")
if(reverse_plot):
    plt.xlim(max(np.log10(prec_list)), min(np.log10(prec_list)))
plt.legend()      
plt.show()
#'''