### QBM TESTS ###
# Run tests to determine how long the QBM takes to learn for certain parameter combinations

import numpy as np
from scipy.linalg import expm
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser!
from time import time
from datetime import datetime
from QBM_Main import QBM, create_H

# Directory to save data to
filedir = "./Data"

#%%% TESTS %%%

# Test parameters
no_inits = 10 # Number of different initializations to average over (to reduce randomness in efficiency due to the initialization)
n_list = [8] # Which qubit amounts to try
optimizer_list = ['GD', 'Nesterov_Book']#, 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR'] # Which optimizers to try

# Magnetic ratios to try for various models (ratio = 1 means random weights should be chosen)
ratio_list = {'Uniform Ising model': [0.2, 0.5, 0.8, 0.9, 0.99, 1, 1.01, 1.1, 1.2, 1.5, 2],
              'Random Ising model': [1, 1, 1, 1, 1]}

# QBM parameters
beta = 1
kmin = 3
precision = 1e-6   # WARNING: A high target precision may make the testing process take very long!
max_qbm_it = 10000
q = 1e-3           # Tuned for uniform ising model with J/h < 1, n <= 6
alpha0 = np.sqrt(q)

### RUN TESTS & SAVE DATA USING h5py ###
# Don't forget to change filedir in the imports section!
f = h5py.File(filedir + '/Data_n8firsthalf_1e-6.hdf5','w')
eps_file = h5py.File(filedir + '/Eps_Data_short_modified.hdf5','r') # File containing best epsilon values

#Unif_Ising_group = f.create_group("Uniform Ising model")
Random_Ising_group = f.create_group("Random Ising model")

# WARNING: Tests for large n take very long! 
# This is heavily dependent on intialization, but for n = 8, runs can already up to 90 minutes per Hamiltonian - Optimizer combination...

total_time_start = time()
print("---------- Commencing Tests on " + datetime.fromtimestamp(total_time_start).strftime("%d-%m-%Y %H:%M:%S") + " ----------")

for key, model_group in f.items():
    if key == "Uniform Ising model":
        uniform_weights = True
    else:
        uniform_weights = False
    
    for n in n_list:
        print("Starting n = {}".format(n)) # DEBUG
        n_group = model_group.create_group("n = {}".format(n))
        n_group.attrs['n'] = n
        H_counter = 1
        
        for ratio in ratio_list[key]:
            # Create Hamiltonian & train QBM
            H, H_params_w, H_params_b = create_H(n, uniform_weights, ratio)
            eta = expm(-beta*H)/expm(-beta*H).trace()
            My_QBM = QBM(eta, n, beta)
            seed_list = np.random.randint(1, 100000, no_inits)
            
            opt_loss = beta * (eta @ H).trace() + np.log(expm(-beta * H).trace()) # Loss for exact parameters (minimum loss)
            
            print("\t Starting Hamiltonian {}".format(H_counter)) # DEBUG
            H_group = n_group.create_group("Hamiltonian {}".format(H_counter))
            H_group.attrs['H_params_w'] = H_params_w
            H_group.attrs['H_params_b'] = H_params_b
            
            for optimizer in optimizer_list:
                print("\t\t Starting optimizer {}".format(optimizer)) # DEBUG
                start_time = time() # To time how long execution takes
                opt_group= H_group.create_group(optimizer)
                opt_group.attrs['Optimizer'] = optimizer
                
                # Average the number of iterations over different initializations
                avg_iterations = []
                first_loop = True
                epsilon = eps_file["{}/n = {}/{}/Best epsilon".format(key, n, optimizer)][()]
                
                for seed in seed_list:
                    np.random.seed(seed)
                    My_QBM.learn(optimizer=optimizer, q=q, alpha0=alpha0, kmin=kmin, max_qbm_it=max_qbm_it, precision=precision, epsilon=epsilon, track_all=False)
                    avg_iterations.append(My_QBM.qbm_it)
                    if first_loop: # Also store the loss per iteration for one seed (for graphing)
                        loss = opt_group.create_dataset('Model loss - Optimal loss', data=My_QBM.loss_QBM_track-opt_loss)
                        first_loop = False
                        
                iters = opt_group.create_dataset('Total iterations', data=np.average(avg_iterations))
                end_time = time()
                print("\t\t Finished on " + datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(end_time - start_time) + " seconds, taking " + str(np.average(avg_iterations)) + " iterations on average")

            H_counter += 1

total_time_end = time()
print("---------- Tests finished on " + datetime.fromtimestamp(total_time_end).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(total_time_end - total_time_start) + " seconds ----------")

f.close()
eps_file.close()