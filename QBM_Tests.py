### QBM TESTS ###
# Run tests to find out how long the QBM takes to learn for certain parameter combinations

import numpy as np
from scipy.linalg import expm
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from time import time
from datetime import datetime
from QBM_Main import QBM, create_H, filedir

#%%% TESTS %%%

# Test parameters (EDIT HERE)
model_list = ["Uniform Ising model"]                               # Which models to try
# Currently best step size values have only been tuned for Random Ising model (uniform is still in progress), so don't try Uniform Ising yet!
n_list = [8]                                                                                # Which qubit amounts to try. WARNING: Tests for large n (8 and up) can take very long!
optimizer_list = ['Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']                             # Which optimizers to try
ratio_list = {'Uniform Ising model': [0.2, 0.5, 0.8, 0.9, 0.99, 1, 1.01, 1.1, 1.2, 1.5, 2],
              'Random Ising model': [1, 1, 1, 1, 1]}                                        # J/h ratios to try (this only matters for the Uniform Ising model)
no_inits = 10                                                                               # Number of different initializations to average over (to reduce randomness due to initialization)

# QBM parameters (EDIT HERE)
beta = 1            # Inverse Boltzmann temperature
kmin = 3            # For Nesterov_SR: minimum number of iterations between restarts
precision = 1e-4    # Precision: the QBM is done learning when the norm of the weight update is smaller than this number. 
                    # WARNING: A high target precision may make the testing process take very long!
max_qbm_it = 10000 # Max number of iterations before the learning process is cut off
q = 1e-3            # For Nesterov_Book: used in calculating the momentum parameter
alpha0 = np.sqrt(q) # For Nesterov_Book: used in calculating the momentum parameter

# Open files
f = h5py.File(filedir + '/Data_n8secondhalf_1e-4_unif.hdf5','w')
eps_file = h5py.File(filedir + '/Eps_Data_short_modified.hdf5','r') # File containing best epsilon values

### RUN TESTS & SAVE DATA USING h5py ###
total_time_start = time()
print("---------- Commencing Tests on " + datetime.fromtimestamp(total_time_start).strftime("%d-%m-%Y %H:%M:%S") + " ----------")

try:
    for model in model_list:
        print("### {} ###".format(model))
        # Read model group, or create it if it doesn't exist
        try:
            model_group = f[model]
        except:
            model_group = f.create_group(model)

        uniform_weights = (model == "Uniform Ising model")
        
        for n in n_list:
            print("Starting n = {}".format(n))
            # Read n group, or create it if it doesn't exist
            try:
                n_group = model_group["n = {}".format(n)]
            except:
                n_group = model_group.create_group("n = {}".format(n))
                n_group.attrs['n'] = n

            H_counter = 1 # Keeps track of how many Hamiltonians we've tested for this n. For the Ising model with uniform weights, this corresponds to different J/h ratios.
            for ratio in ratio_list[model]:
                # Create target Hamiltonian & initialize QBM
                H, H_params_w, H_params_b = create_H(n, uniform_weights, ratio)
                eta = expm(-beta*H)/expm(-beta*H).trace()
                My_QBM = QBM(eta, n, beta)

                seed_list = np.random.randint(1, 100000, no_inits) # Generate randomizer seeds; this is to keep initializations consistent between optimizers.
                
                opt_loss = beta * (eta @ H).trace() + np.log(expm(-beta * H).trace()) # Loss for exact parameters (minimum loss)
                
                print("\t Starting Hamiltonian {}".format(H_counter))
                # Read Hamiltonian group, or create it if it doesn't exist
                try:
                    H_group = n_group["Hamiltonian {}".format(H_counter)]  
                except:
                    H_group = n_group.create_group("Hamiltonian {}".format(H_counter))
                    H_group.attrs['H_params_w'] = H_params_w
                    H_group.attrs['H_params_b'] = H_params_b
                    H_group.attrs['J/h ratio'] = ratio            

                for optimizer in optimizer_list:
                    print("\t\t Starting optimizer {}".format(optimizer))
                    # Read optimizer group, or create it if it doesn't exist
                    try:
                        opt_group = H_group[optimizer]
                        # If optimizer group already exists, check if it already has data
                        try:
                            test = opt_group['Total iterations'][()]
                        except: # If not, continue testing
                            print("\t\t Optimizer group already exists, but has no data. Commencing testing")
                        else:   # If data does exist, move on to next optimizer
                            print("\t\t Optimizer group already exists with data, moving on to next optimizer")
                            continue
                    except:
                        opt_group = H_group.create_group(optimizer)
                        opt_group.attrs['Optimizer'] = optimizer
                        
                    start_time = time() # To time how long execution takes
                    
                    avg_iterations = [] # Stores total iterations per seed, to average over later
                    first_loop = True
                    epsilon = eps_file["{}/n = {}/{}/Best epsilon".format(model, n, optimizer)][()]
                    
                    for seed in seed_list:
                        # Train the QBM & store how many iterations it takes
                        np.random.seed(seed)
                        My_QBM.learn(optimizer=optimizer, q=q, alpha0=alpha0, kmin=kmin, max_qbm_it=max_qbm_it, precision=precision, epsilon=epsilon, track_all=False)
                        avg_iterations.append(My_QBM.qbm_it)
                        # Also store the loss per iteration for one seed (for plotting purposes)
                        if first_loop: 
                            loss = opt_group.create_dataset('Model loss - Optimal loss', data=My_QBM.loss_QBM_track-opt_loss)
                            first_loop = False
                            
                    # Average the number of iterations over different initializations & store it in the data file
                    iters = opt_group.create_dataset('Total iterations', data=np.average(avg_iterations))
                    end_time = time()
                    print("\t\t Finished on " + datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(end_time - start_time) + " seconds, taking " + str(np.average(avg_iterations)) + " iterations on average")

                H_counter += 1 # Move on to next H

    total_time_end = time()
    print("---------- Tests finished on " + datetime.fromtimestamp(total_time_end).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(total_time_end - total_time_start) + " seconds ----------")

# Even if testing is interrupted, make sure the files are properly closed
except KeyboardInterrupt:
    print("Interrupted by user!")
finally:
    f.close()
    eps_file.close()