### HYPERPARAMETER TUNING ###
# Run tests to determine optimal values for hyperparameters (e.g. stepsize)

import numpy as np
from scipy.linalg import expm
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from time import time
from datetime import datetime
from QBM_Main import QBM, create_H, filedir

#%%% FIND OPTIMAL STEPSIZE %%%
# Try various stepsizes for various Hamiltonians of a model class and see which works best.
# We store the best stepsize per (model, n, optimizer) combination, averaging over different Hamiltonians (J/h ratios) and initializations.
# This code can also be used as a base for tuning other hyperparameters (e.g. kmin or q); 
# in that case, replace eps_list with whichever parameter you want to tune and change names accordingly.

# Test parameters (EDIT HERE)
eps_list= [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]                                       # Which stepsizes to try
model_list = ["Random Ising model", "Uniform Ising model"]                               # Which models to try
n_list = [2,4,6,8]                                                                          # Which qubit amounts to try
optimizer_list = ['Adam']      # Which optimizers to try
ratio_list = {'Uniform Ising model': [0.2, 0.5, 0.8, 0.9, 0.99, 1, 1.01, 1.1, 1.2, 1.5, 2],
              'Random Ising model': [1, 1, 1, 1, 1]}                                        # J/h ratios to average over (this only matters for the Uniform Ising model)
no_inits = 10                                                                               # Number of different initializations to average over (to reduce randomness in efficiency due to the initialization)

# QBM parameters (EDIT HERE)
beta = 1            # Inverse Boltzmann temperature
kmin = 3            # For Nesterov_SR: minimum number of iterations between restarts
precision = 1e-4    # Precision: the QBM is done learning when the norm of the weight update is smaller than this number. 
                    # WARNING: A high target precision may make the tuning process take very long!
max_qbm_it = 10000  # Max number of iterations before the learning process is cut off
q = 1e-3            # For Nesterov_Book: used in calculating the momentum parameter
alpha0 = np.sqrt(q) # For Nesterov_Book: used in calculating the momentum parameter

f = h5py.File(filedir + '/Eps_Data_Adam.hdf5','a')

total_time_start = time()
print("---------- Commencing Tuning on " + datetime.fromtimestamp(total_time_start).strftime("%d-%m-%Y %H:%M:%S") + " ----------")

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

            for optimizer in optimizer_list:
                print("\t Starting optimizer {}".format(optimizer))
                # Read optimizer group, or create it if it doesn't exist
                try:
                    opt_group = n_group[optimizer]
                except:
                    opt_group = n_group.create_group(optimizer)
                    opt_group.attrs['Optimizer'] = optimizer

                rand_seed = np.random.randint(1, 100000) # To ensure each epsilon has the same initialization
                iters_per_eps = [] # Keep track of how many iterations are taken for each epsilon
                first_eps = True

                for epsilon in eps_list:
                    print("\t\t Starting epsilon = {}".format(epsilon)) # DEBUG
                    np.random.seed(rand_seed)
                    start_time = time() # To time how long execution takes
                    # Read epsilon group, or create it if it doesn't exist
                    try:
                        eps_group = opt_group["epsilon = {}".format(epsilon)]
                        # If epsilon group already exists, check if it already has data
                        try:
                            test = eps_group['Total iterations'][()]
                        except: # If not, continue testing
                            print("\t\t Epsilon group already exists, but has no data. Commencing testing")
                        else:   # If data does exist, move on to next epsilon
                            print("\t\t Epsilon group already exists with data, moving on to next epsilon")
                            iters_per_eps.append(test)
                            continue
                    except:
                        eps_group = opt_group.create_group("epsilon = {}".format(epsilon))
                        eps_group.attrs['epsilon'] = epsilon
                    
                    avg_iterations = [] # Stores total iterations per Hamiltonian & seed, to average over later

                    for ratio in ratio_list[model]:
                        # Create Hamiltonian & initialize QBM
                        H, H_params_w, H_params_b = create_H(n, uniform_weights, ratio)
                        eta = expm(-beta*H)/expm(-beta*H).trace()
                        My_QBM = QBM(eta, n, beta)

                        # Train QBM for various initializations
                        for run in range(no_inits):
                            My_QBM.learn(optimizer=optimizer, q=q, alpha0=alpha0, kmin=kmin, max_qbm_it=max_qbm_it, precision=precision, epsilon=epsilon, track_all=False)
                            avg_iterations.append(My_QBM.qbm_it) 

                    # Average the number of iterations over different initializations & store it in the data file
                    avg = np.average(avg_iterations)
                    iters = eps_group.create_dataset('Total iterations', data=avg) # Average iterations for this (model, n, optimizer, epsilon) combo
                    iters_per_eps.append(avg)

                    end_time = time()
                    print("\t\t Finished on " + datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(end_time - start_time) + " seconds, taking " + str(np.average(avg_iterations)) + " iterations on average")
                    
                    # Stop trying new epsilons if the number of iterations has gone up since the last time (we've overshot the optimal epsilon)
                    if(not(first_eps) and avg > iters_per_eps[-2]):
                        print("\t\t Iterations have gone up, moving on to next optimizer")
                        break
                    first_eps = False

                # Store the best epsilon for this (model, n, optimizer) combo in the data file
                best_eps = eps_list[np.argmin(iters_per_eps)]
                opt_group.create_dataset('Best epsilon', data=best_eps)
                print("\t Best epsilon: {}".format(best_eps))  

    total_time_end = time()
    print("---------- Tuning finished on " + datetime.fromtimestamp(total_time_end).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(total_time_end - total_time_start) + " seconds ----------")

# Even if testing is interrupted, make sure the files are properly closed
except KeyboardInterrupt:
    print("Interrupted by user!")
finally:
    f.close()