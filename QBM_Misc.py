### MISCELLANEOUS ###
# Miscellaneous/old code snippets and tests for reference. Not very well documented.
# WARNING: These may apply to a previous version of the code, and as such, might not work with the current structure!

import numpy as np
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from QBM_Main import filedir

# Close data files after testing
#f.close()
#eps_file.close()

'''
# Test
eps_file = h5py.File(filedir + '\\Eps_Data.hdf5','r')

key = "Random Ising model"
n = 2
optimizer = 'GD'

print(eps_file["{}/n = {}/{}/Best epsilon".format(key, n, optimizer)][()])

eps_file.close()

# CODE SNIPPET FOR PRINTING ALL h5py DATA
for key, _ in f.items():
    for n in n_list:
        n_group = f["{}/n = {}".format(key,n)]
        print("n = {}".format(n_group.attrs['n']))
        H_counter = 1
        
        for ratio in ratio_list:
            H_group = n_group["Hamiltonian {}".format(H_counter)]
            print('\t' + "Hamiltonian params: ")
            print(H_group.attrs['H_params_w'])
            print(H_group.attrs['H_params_b'])
            
            for optimizer in optimizer_list:
                opt_group= H_group[optimizer]
                print('\t\t' + "Optimizer: {}".format(opt_group.attrs['Optimizer']))
                
                loss = opt_group["Loss"][()]
                iters = opt_group["Total iterations"][()]
                #print('\t\t\t' + "First 10 loss values: ")
                #print(loss[0:10])
                print('\t\t\t' + "Total iterations: {}".format(iters))

            H_counter += 1
'''
'''
#%% MODIFY NESTEROV EPSILONS
# It appears something went wrong when tuning stepsize for n = 4, which caused the "optimal" step size to be too large in most cases.
# To avoid having to re-run the code (which would likely take a day or two), I'll manually decrease all optimal step sizes to one value lower using this snippet.

eps_file_complete_random = h5py.File(filedir + '/Eps_Data_complete.hdf5','r') 
eps_file_short_random = h5py.File(filedir + '/Eps_Data_short_random.hdf5','r') 

eps_file_short_modified = h5py.File(filedir + '/Eps_Data_short_modified.hdf5','w')

for n in [2,4,6,8]:
    for optimizer in ['GD','Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']:
        best_eps = eps_file_short_random['Random Ising model/n = {}/{}/Best epsilon'.format(n, optimizer)][()]
        if n == 4 and optimizer in ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']:
            best_eps = 0.2
        opt_group_s = eps_file_short_modified.create_group('Random Ising model/n = {}/{}'.format(n, optimizer))
        opt_group_s.create_dataset('Best epsilon', data=best_eps)
        
eps_file_complete_random.close()
eps_file_short_random.close()
eps_file_short_modified.close()
'''

'''
# Check which optimizers exhibit sawtooth-behavior (useful to check where stepsize is too large)
f = h5py.File(filedir + '/Data_random_until1e-5.hdf5','r')

for precision in ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]:
    for H in [1,2,3,4,5]:
        for optimizer in ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']:
            n = 4
            path = 'Random Ising model/precision = {}/n = {}/Hamiltonian {}/{}/Total iterations'.format(precision, n, H, optimizer)
            iters = f[path][()]
            if iters == 10000:
                print("10000 iterations reached for n = 4, optimizer " + optimizer)

f.close()
'''
'''
#%%% MERGE EPSILON FILES %%%
# Example of how to merge epsilon files for different models

# Source files (EDIT HERE)
file1 = h5py.File(filedir + '/Eps_Data_uniform.hdf5','r')      # Uniform Ising model
file2 = h5py.File(filedir + '/Eps_Data_complete_random.hdf5','r')  # Random Ising model

# Target files (EDIT HERE)
file_target = h5py.File(filedir + '/Eps_Data_short.hdf5','w') # File to write all data to

# Parameters to merge (EDIT HERE)
model_list = ["Uniform Ising model", "Random Ising model"]
n_list = [2,4,6,8]
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
merge_all = False  # Whether to merge all data (best epsilon and average iterations per epsilon) or just the best epsilon for each (model, n, optimizer) combo

# Merge data
for model in model_list:
    for n in n_list:
        for optimizer in optimizer_list:
            path = "{}/n = {}/{}".format(model, n, optimizer)
            opt_group = file_target.create_group(path)

            # Select the right file to read
            if model == "Uniform Ising model":
                read_file = file1
            else:
                read_file = file2

            # Merge best epsilon
            best_eps = read_file[path + '/Best epsilon'][()]
            opt_group.create_dataset('Best epsilon', data=best_eps)

            # Merge average iterations per epsilon (for each epsilon that was tested)
            if merge_all:
                for epsilon in eps_list[0:eps_list.index(best_eps)+2]:
                    eps_group = opt_group.create_group("epsilon = {}".format(epsilon))
                    eps_group.create_dataset('Total iterations', data=read_file[path + '/epsilon = {}/Total iterations'.format(epsilon)][()])
    
file1.close()
file2.close()
file_target.close()
'''
'''
#%% MODIFY NESTEROV EPSILONS
# It appears something went wrong when tuning stepsize for n = 4, which caused the "optimal" step size to be too large in most cases.
# To avoid having to re-run the code (which would likely take a day or two), I'll manually decrease all optimal step sizes to one value lower using this snippet.

# Source files (EDIT HERE)
file1 = h5py.File(filedir + '/Eps_Data_short.hdf5','r')

# Target files (EDIT HERE)
file_target = h5py.File(filedir + '/Eps_Data_short_modified.hdf5','w') # File to write all data to

# Parameters to merge (EDIT HERE)
model_list = ["Uniform Ising model", "Random Ising model"]
n_list = [2,4,6,8]
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
merge_all = False  # Whether to merge all data (best epsilon and average iterations per epsilon) or just the best epsilon for each (model, n, optimizer) combo

# Merge data
for model in model_list:
    for n in n_list:
        for optimizer in optimizer_list:
            path = "{}/n = {}/{}".format(model, n, optimizer)
            opt_group = file_target.create_group(path)

            read_file = file1

            # Merge best epsilon
            best_eps = read_file[path + '/Best epsilon'][()]
            if n == 4 and optimizer in ['GD', 'Nesterov_Book']:
                opt_group.create_dataset('Best epsilon', data=eps_list[eps_list.index(best_eps)-1])
            else:
                opt_group.create_dataset('Best epsilon', data=best_eps)

            # Merge average iterations per epsilon (for each epsilon that was tested)
            if merge_all:
                for epsilon in eps_list[0:eps_list.index(best_eps)+2]:
                    eps_group = opt_group.create_group("epsilon = {}".format(epsilon))
                    eps_group.create_dataset('Total iterations', data=read_file[path + '/epsilon = {}/Total iterations'.format(epsilon)][()])
    
file1.close()
file_target.close()
'''