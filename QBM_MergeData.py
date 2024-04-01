### MERGE DATA FILES ###
# Some code for merging different data files together (when testing has to be split into batches)
# Uncomment the script you want to run (may have to split this into different files or find some way to make this more convenient later)

import numpy as np
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from QBM_Main import filedir
#'''
#%%% MERGE DATA FILES TOGETHER %%%
# Example of how to merge data files when testing is split into 3 runs: all optimizer until n = 6, GD and Nesterov_Book for n = 8, and Nesterov_SBC, Nesterov_GR and Nesterov_SR for n = 8
# All these tests are executed for the same precision, in this case 1e-5
#'''
# Source files (EDIT HERE)
file1 = h5py.File(filedir + '/Data_untiln6_1e-6.hdf5','r')      # All optimizers until n = 6
file2 = h5py.File(filedir + '/Data_n8firsthalf_1e-6.hdf5','r')  # n = 8, GD and Nesterov_Book
file3 = h5py.File(filedir + '/Data_n8secondhalf_1e-6.hdf5','r') # n = 8, Nesterov_SBC, Nesterov_GR and Nesterov_SR

# Target files (EDIT HERE)
file_target = h5py.File(filedir + '/Data_random_1e-6.hdf5','w') # File to write all data to
file_target = h5py.File(filedir + '/Data_random_1e-6.hdf5','w') # File to write all data to

# Parameters to merge (EDIT HERE)
model = "Random Ising model"
n_list = [2,4,6,8]
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
H_count = 5 # Number of Hamiltonians that were tested for the given model (11 for uniform Ising, 5 for random Ising)

# Merge data
for n in n_list:
    for H_counter in range(1, H_count+1): 
        for optimizer in optimizer_list:
            path = "{}/n = {}/Hamiltonian {}/{}".format(model, n, H_counter, optimizer)
            opt_group = file_target.create_group(path)
            
            # Select the right file to read
            if n < 8:
                read_file = file1
            elif optimizer in ['GD', 'Nesterov_Book']:
                read_file = file2
            else:
                read_file = file3
            
            opt_group.create_dataset('Model loss - Optimal loss', data = read_file[path + '/Model loss - Optimal loss'][()])
            opt_group.create_dataset('Total iterations', data = read_file[path + '/Total iterations'][()])
   
file1.close()
file2.close()
file3.close()
file_target.close()
#'''

'''
#%%% MERGE DATA FILES FOR DIFFERENT PRECISIONS %%%
# Example of how to merge complete data files (made using the above script) for different precisions into one big file

# Source files (EDIT HERE)
prec_list = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
source_files = []
for prec in prec_list:
    source_files.append(h5py.File(filedir + '/Data_random_' + prec + '.hdf5','r'))

# Target files (EDIT HERE)
file_target = h5py.File(filedir + '/Data_random_until1e-5.hdf5','w') # File to write all data to

# PARAMETERS (EDIT HERE)
model = "Random Ising model"
n_list = [2,4,6,8]
optimizer_list = ['GD','Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
H_count = 5 # Number of Hamiltonians that were tested for the given model (11 for uniform Ising, 5 for random Ising)

# MERGE DATA
for i in range(len(prec_list)):
    for n in n_list:
        for H_counter in range(1, H_count+1): 
            for optimizer in optimizer_list:
                path = "{}/n = {}/Hamiltonian {}/{}".format(model, n, H_counter, optimizer)
                opt_group = file_target.create_group("{}/precision = {}/n = {}/Hamiltonian {}/{}".format(model, prec_list[i], n, H_counter, optimizer))
                
                read_file = source_files[i]
                
                opt_group.create_dataset('Model loss - Optimal loss', data = read_file[path + '/Model loss - Optimal loss'][()])
                opt_group.create_dataset('Total iterations', data = read_file[path + '/Total iterations'][()])
   
for i in range(len(prec_list)):
    source_files[i].close()

file_target.close()
'''

'''
#%%% MERGE EPSILON FILES %%%
# Example of how to merge different epsilon files (created using the QBM_TuneParams script), similar to the first script

# Source files (EDIT HERE)
file1 = h5py.File(filedir + '/_Eps_Data_untiln6.hdf5','r')      # All optimizers until n = 6
file2 = h5py.File(filedir + '/_Eps_Data_n8firsthalf.hdf5','r')  # n = 8, GD and Nesterov_Book
file3 = h5py.File(filedir + '/_Eps_Data_n8secondhalf.hdf5','r') # n = 8, Nesterov_SBC, Nesterov_GR and Nesterov_SR

# Target files (EDIT HERE)
file_target = h5py.File(filedir + '/Eps_Data_uniform.hdf5','w') # File to write all data to

# Parameters to merge (EDIT HERE)
model = "Uniform Ising model"
n_list = [2,4,6,8]
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
merge_all = True  # Whether to merge all data (best epsilon and average iterations per epsilon) or just the best epsilon for each (model, n, optimizer) combo

# Merge data
for n in n_list:
    for optimizer in optimizer_list:
        path = "{}/n = {}/{}".format(model, n, optimizer)
        opt_group = file_target.create_group(path)

        # Select the right file to read
        if n < 8:
            read_file = file1
        elif optimizer in ['GD', 'Nesterov_Book']:
            read_file = file2
        else:
            read_file = file3

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
file3.close()
file_target.close()
'''