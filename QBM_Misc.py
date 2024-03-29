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

eps_file_complete_random = h5py.File(filedir + '/Eps_Data_complete_random.hdf5','r') # File to write all data to
eps_file_short_random = h5py.File(filedir + '/Eps_Data_short_random.hdf5','r') # File to write only the best epsilon per (model, n, optimizer) combo to

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