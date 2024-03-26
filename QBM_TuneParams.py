### HYPERPARAMETER TUNING ###
# Run tests to determine optimal values for hyperparameters (e.g. stepsize)

import numpy as np
from scipy.linalg import expm
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser!
from time import time
from datetime import datetime
from QBM_Main import QBM, create_H

# Directory where data should be stored
filedir = "./Data"

#%%% FIND OPTIMAL HYPERPARAMETERS %%%

# Try various stepsizes for various Hamiltonians of a model class and see which works best.
# Iteratively get closer to the minimum.

# Test parameters
no_inits = 10 # Number of different initializations to average over (to reduce randomness in efficiency due to the initialization)
n_list = [2,4,6,8] # Which qubit amounts to try
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR'] # Which optimizers to try
#optimizer_list = ['Nesterov_Book']

# Magnetic ratios to try for various models (ratio = 1 means random weights should be chosen)
ratio_list = {'Uniform Ising model': [0.2, 0.5, 0.8, 0.9, 0.99, 1, 1.01, 1.1, 1.2, 1.5, 2],
              'Random Ising model': [1, 1, 1, 1, 1]}

# QBM parameters
beta = 1
kmin = 3
precision = 1e-4   # WARNING: A high target precision may make the testing process take very long!
max_qbm_it = 10000
#epsilon = 0.25
q = 1e-3
alpha0 = np.sqrt(q)

# Tune optimal step size
eps_list= [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#iters_per_eps = dict(zip(eps_list, np.zeros(len(eps_list))))

# Nesterov_SR: Tune optimal kmin 
#kmin_list = [1, 2, 3, 4, 5]
#iters_per_kmin = dict(zip(kmin_list, np.zeros(len(kmin_list))))

# Nesterov_Book: Tune optimal q (and alpha = sqrt(q))
#q_list = [1e-3, 5e-3, 1e-2]
#iters_per_q = dict(zip(q_list, np.zeros(len(q_list))))

f = h5py.File(filedir + '\\Eps_Data_GDn8.hdf5','w')

#Unif_Ising_group = f.create_group("Uniform Ising model")
Random_Ising_group = f.create_group("Random Ising model")

# WARNING: Tests for large n take very long! 
# This is heavily dependent on intialization, but for n = 8, runs can already up to 90 minutes per Hamiltonian - Optimizer combination...

total_time_start = time()
print("---------- Commencing Tuning on " + datetime.fromtimestamp(total_time_start).strftime("%d-%m-%Y %H:%M:%S") + " ----------")

for key, model_group in f.items():
    if key == "Uniform Ising model":
        uniform_weights = True
    else:
        uniform_weights = False
    
    for n in n_list:
        print("Starting n = {}".format(n)) # DEBUG
        n_group = model_group.create_group("n = {}".format(n))
        n_group.attrs['n'] = n
        
        for optimizer in optimizer_list:
            print("\t Starting optimizer {}".format(optimizer)) # DEBUG
            opt_group= n_group.create_group(optimizer)
            opt_group.attrs['Optimizer'] = optimizer
            
            rand_seed = np.random.randint(1, 100000) # To ensure each epsilon has the same initialization
            
            iters_per_eps = [] # Keep track of how many iterations are taken for each epsilon
            first_eps = True
            
            for epsilon in eps_list:
                print("\t\t Starting epsilon = {}".format(epsilon)) # DEBUG
                np.random.seed(rand_seed)
                start_time = time() # To time how long execution takes
                eps_group = opt_group.create_group("epsilon = {}".format(epsilon))
                eps_group.attrs['epsilon'] = epsilon
                
                H_counter = 1
                
                # Average the number of iterations over different Hamiltonians & initializations
                avg_iterations = []
                
                for ratio in ratio_list[key]:
                    # Create Hamiltonian & train QBM
                    H, H_params_w, H_params_b = create_H(n, uniform_weights, ratio)
                    eta = expm(-beta*H)/expm(-beta*H).trace()
                    My_QBM = QBM(eta, n, beta)
                    
                    for run in range(no_inits):
                        My_QBM.learn(optimizer=optimizer, q=q, alpha0=alpha0, kmin=kmin, max_qbm_it=max_qbm_it, precision=precision, epsilon=epsilon, track_all=False)
                        avg_iterations.append(My_QBM.qbm_it)
                            
                    H_counter += 1
                
                avg = np.average(avg_iterations)
                iters = eps_group.create_dataset('Total iterations', data=avg)
                iters_per_eps.append(avg)
                end_time = time()
                print("\t\t Finished on " + datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(end_time - start_time) + " seconds, taking " + str(np.average(avg_iterations)) + " iterations on average")
                # Stop trying new epsilons if the number of iterations has gone up since the last time
                # (We've overshot the optimal epsilon)
                if(not(first_eps) and avg > iters_per_eps[-2]):
                    print("\t\t Iterations have gone up, moving on to next optimizer")
                    break
                first_eps = False

            # Store the best epsilon for this (model, n, optimizer) combo
            best_eps = eps_list[np.argmin(iters_per_eps)]
            opt_group.create_dataset('Best epsilon', data=best_eps)
            print("\t Best epsilon: {}".format(best_eps))

total_time_end = time()
print("---------- Tuning finished on " + datetime.fromtimestamp(total_time_end).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(total_time_end - total_time_start) + " seconds ----------")

f.close()

# Epsilons for precision = 1e-7:
# n = 2
#   0.5 for all optimizers
# n = 4
#   GD: 0.5, Nesterov_Book: 0.3, Nesterov_SBC: 0.2, Nesterov_GR: 0.3, Nesterov_SR: 0.2
# n = 6
#   GD: 0.2, Nesterov_Book: 0.1

# Epsilons for precision = 1e-4:
# n = 2
#   0.5 for all optimizers
# n = 4
#   GD: 0.4, Nesterov_Book: 0.3, Nesterov_SBC: 0.3, Nesterov_GR: 0.2, Nesterov_SR: 0.3
# n = 6
#   GD: 0.2, Nesterov_Book: 0.1

''' OLD
for epsilon in eps_list: # Change to what hyperparameter you want to tune
    np.random.seed(512)   # Fix initializations for different hyperparams (for better comparison)
    
    total_avg_iters = []
    
    for n in n_list:
        print("Starting n = {}".format(n)) # DEBUG
        H_counter = 1
        
        for ratio in ratio_list['Random Ising model']:
            # Create Hamiltonian & train QBM
            H, H_params_w, H_params_b = create_H(n, False, ratio)
            eta = expm(-beta*H)/expm(-beta*H).trace()
            My_QBM = QBM(eta, n, beta)
            seed_list = np.random.randint(1, 100000, no_inits)
            
            print("\t Starting Hamiltonian {}".format(H_counter)) # DEBUG
            
            for optimizer in optimizer_list:
                print("\t\t Starting optimizer {}".format(optimizer)) # DEBUG
                start_time = time() # To time how long execution takes
                
                # Average the number of iterations over different initializations
                avg_iterations = []
                
                for seed in seed_list:
                    np.random.seed(seed)
                    My_QBM.learn(optimizer=optimizer, q=q, alpha0=np.sqrt(q), kmin=kmin, max_qbm_it=max_qbm_it, precision=precision, epsilon=epsilon, track_all=False)
                    avg_iterations.append(My_QBM.qbm_it)
                        
                total_avg_iters.append(np.average(avg_iterations))
                end_time = time()
                print("\t\t Finished in " + str(end_time - start_time) + " seconds, taking " + str(np.average(avg_iterations)) + " iterations on average")

            H_counter += 1
            
    print("\n --- Average number of iterations for epsilon = {}: {} ---".format(epsilon, np.average(total_avg_iters)))
    iters_per_eps[epsilon] = np.average(total_avg_iters)

print("Average iterations per epsilon: ")
print(iters_per_eps)

# Average iterations per epsilon: 
# {0.08: 3042.335, 0.1: 2949.69, 0.15: 2770.018333333334, 0.2: 2673.708333333333}
# {0.2: 2673.708333333333, 0.3: 3075.1483333333335, 0.4: 4084.45}
# I forgot to copy-paste some of the test results from my console (stupid),
# but the best epsilon turned out to be about 0.25 (precise up to 2 digits)

# WATCH OUT: Epsilon HAS to be tuned per optimizer! I spent hours trying to figure out why the Nesterov_Book optimizer
# was giving "zig-zaggy" behavior, and it turns out it was overstepping.
# It seems *some* optimizers can get away with larger step sizes, but making it too large 
# can prevent other optimizers from ever learning.
# My suggestion is we tune stepsize for the combination of n, optimizer, and model type.

# These are typical numbers:
# {0.24: 1984.4816666666668, 0.25: 1959.2000000000003, 0.26: 1960.0766666666668, 0.27: 1947.9416666666666, 0.28: 1900.855}
# Which of these ends up actually being the lowest is dependent on the initial seed, but the minimum is somewhere around here.

#Average iterations per kmin: 
#{20: 575.26, 50: 653.5799999999999, 100: 835.7199999999999, 200: 845.5666666666668, 300: 1125.6533333333332, 500: 1232.68}
#{1: 504.5266666666667, 2: 507.0866666666667, 5: 510.90666666666664, 10: 526.76, 15: 534.0666666666667, 20: 575.26}
# Different seed: {1: 1448.8733333333332, 2: 1449.9933333333333, 3: 1445.2066666666665, 4: 1445.8866666666665, 5: 1452.4}
# Yet another seed: {1: 420.30000000000007, 2: 415.66, 3: 416.04, 4: 416.1133333333334, 5: 420.9266666666666}
# These are all very close together, but I'd say kmin = 3 is generally a good bet.

#Average iterations per q: 
#{0.0001: 2957.92, 0.001: 2823.713333333333, 0.01: 2785.1200000000003, 0.1: 2936.9}
# Different seed: {0.001: 325.1466666666667, 0.005: 375.22666666666663, 0.01: 425.87333333333333, 0.025: 546.2000000000002, 0.05: 687.7600000000001}
#Yet another seed: {0.001: 1796.9266666666667, 0.005: 1946.2400000000005, 0.01: 2081.4666666666667}
# It seems q = 0.001 gives the best results in general.
'''