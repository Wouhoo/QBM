### MERGE DATA FILES ###
# Some code for merging different data files together (when testing has to be split into batches)

import numpy as np
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser!

# Directory where data may be found
filedir = "./Data"

#%%% MERGE (EPS_)DATA FILES TOGETHER %%%

# SOURCE FILES (EDIT HERE)
file1 = h5py.File(filedir + '\\Data_untiln6_1e-5.hdf5','r')      # All optimizers until n = 6
file2 = h5py.File(filedir + '\\Data_n8firsthalf_1e-5.hdf5','r')  # n = 8, GD and Nesterov_Book
file3 = h5py.File(filedir + '\\Data_n8secondhalf_1e-5.hdf5','r') # n = 8, Nesterov_SBC, Nesterov_GR and Nesterov_SR

# TARGET FILES (EDIT HERE)
file_target = h5py.File(filedir + '\\Data_random_1e-5.hdf5','w') # File to write all data to
#file_target_short = h5py.File(filedir + '\\Eps_Data_short_random.hdf5','a') # File to write only the best epsilon per (model, n, optimizer) combo to

# PARAMETERS (EDIT HERE)
model = "Random Ising model"
n_list = [2,4,6,8]
optimizer_list = ['GD','Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']
H_count = 5 # Number of Hamiltonians that were tested for the given model (11 for uniform Ising, 5 for random Ising)

# MERGE DATA
for n in n_list:
    for H_counter in range(1, H_count+1): 
        for optimizer in optimizer_list:
            path = "{}/n = {}/Hamiltonian {}/{}".format(model, n, H_counter, optimizer)
            opt_group= file_target.create_group(path)
            
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

'''
# MERGE EPSILON FILES
for model_group in [Random_Ising_group_c]:
    for n in n_list:
        n_group = model_group["n = {}".format(n)]
        
        for optimizer in optimizer_list:
            opt_group= n_group.create_group(optimizer)
            opt_group.attrs['Optimizer'] = optimizer

            best_eps = eps_file2['Random Ising model/n = {}/{}/Best epsilon'.format(n, optimizer)][()]
            opt_group.create_dataset('Best epsilon', data=best_eps)
            
            opt_group_s = eps_file_short_random.create_group('Random Ising model/n = {}/{}'.format(n, optimizer))
            opt_group_s.create_dataset('Best epsilon', data=best_eps)
            
            for epsilon in eps_list[0:eps_list.index(best_eps)+2]:
                eps_group = opt_group.create_group("epsilon = {}".format(epsilon))
                eps_group.attrs['epsilon'] = epsilon
                eps_group.create_dataset('Total iterations', data=eps_file2["Random Ising model/n = {}/{}/epsilon = {}/Total iterations".format(n, optimizer, epsilon)][()])
   
eps_file1.close()
eps_file2.close()
eps_file3.close()
eps_file_complete_random.close()
eps_file_short_random.close()
'''
'''
#%% MERGE DATA FILES FOR DIFFERENT PRECISIONS

# SOURCE FILES (EDIT HERE)
prec_list = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
source_files = []
for prec in prec_list:
    source_files.append(h5py.File(filedir + '\\Data_random_' + prec + '.hdf5','r'))

# TARGET FILES (EDIT HERE)
file_target = h5py.File(filedir + '\\Data_random_until1e-5.hdf5','w') # File to write all data to
#file_target_short = h5py.File(filedir + '\\Eps_Data_short_random.hdf5','a') # File to write only the best epsilon per (model, n, optimizer) combo to

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
                opt_group= file_target.create_group("{}/precision = {}/n = {}/Hamiltonian {}/{}".format(model, prec_list[i], n, H_counter, optimizer))
                
                read_file = source_files[i]
                
                opt_group.create_dataset('Model loss - Optimal loss', data = read_file[path + '/Model loss - Optimal loss'][()])
                opt_group.create_dataset('Total iterations', data = read_file[path + '/Total iterations'][()])
   
for i in range(len(prec_list)):
    source_files[i].close()

file_target.close()
'''