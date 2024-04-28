### QBM TESTS WITH SALAMANDER DATA ###
# Run tests to find out how long the QBM takes to learn selections from the Salamander Retina dataset

import numpy as np
from scipy.linalg import expm
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from time import time
from datetime import datetime
from QBM_Main import QBM, create_H, create_H_from_param, filedir
from DATA_Salamander import give_salamander_dens_matrix

#%%% TESTS %%%

# Test parameters (EDIT HERE)
n_list = [2,4,6,8]                                                                              # Which qubit amounts to try. WARNING: Tests for large n (8 and up) can take very long!
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR', 'Adam']  # Which optimizers to try
no_runs = 3                                                                                     # Number of different selections to take from the salamander dataset for each n
no_inits = 10                                                                                   # Number of different initializations to average over (to reduce randomness due to initialization)

# QBM parameters (EDIT HERE)
beta = 1                                         # Inverse Boltzmann temperature
prec_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] # List of target precisions (we track how many iterations the QBM takes to reach each of these)
max_qbm_it = 1000000                             # Max number of iterations before the learning process is cut off
kmin = 3                                         # For Nesterov_SR: minimum number of iterations between restarts
q = 1e-3                                         # For Nesterov_Book: used in calculating the momentum parameter
alpha0 = np.sqrt(q)                              # For Nesterov_Book: used in calculating the momentum parameter

# Open files
f = h5py.File(filedir + '/Data_QBM_Salamander.hdf5','a') # File to write data to
eps_file = h5py.File(filedir + '/Eps_Data_short_modified.hdf5','r') # File containing best stepsize values

### RUN TESTS & SAVE DATA USING h5py ###
### WARNING: TESTS MAY NOT BE INTERRUPTED WHILE A RUN IS GOING ON ###
total_time_start = time()
print("---------- Commencing Tests on " + datetime.fromtimestamp(total_time_start).strftime("%d-%m-%Y %H:%M:%S") + " ----------")

try:
    for n in n_list:
        print("Starting n = {}".format(n))
        # Read n group, or create it if it doesn't exist
        try:
            n_group = f["n = {}".format(n)]
        except:
            n_group = f.create_group("n = {}".format(n))
            n_group.attrs['n'] = n

        for run in range(1,no_runs+1):
            # Read run group, or create it if it doesn't exist
            print("\t Starting run {}".format(run))
            try:
                run_group = n_group["Run {}".format(run)]  
                print("Already started this run for this n.")
            except:
                run_group = n_group.create_group("Run {}".format(run))

            # Get salamander data
            salamander_data = give_salamander_dens_matrix(n)
            eta = salamander_data['dens_matrix']
            run_group.attrs['experiment_number'] = salamander_data['experiment_number']
            run_group.attrs['selected_neurons'] = salamander_data['selected_neurons']

            # Initialize QBM
            My_QBM = QBM(eta, n, beta)

            seed_list = np.random.randint(1, 100000, no_inits) # Generate randomizer seeds; this is to keep initializations consistent between optimizers.

            for optimizer in optimizer_list:
                print("\t\t Starting optimizer {}".format(optimizer))
                # Read optimizer group, or create it if it doesn't exist
                try:
                    opt_group = run_group[optimizer]
                    # If optimizer group already exists, check if it already has data
                    try:
                        test = opt_group['Total iterations to reach prec_list precisions'][()]
                    except: # If not, continue testing
                        print("\t\t Optimizer group already exists, but has no data. Commencing testing")
                    else:   # If data does exist, move on to next optimizer
                        print("\t\t Optimizer group already exists with data, moving on to next optimizer.")
                        print("\t\t WARNING: Next optimizer will use a different selection of salamander data!")
                        continue
                except:
                    opt_group = run_group.create_group(optimizer)
                    opt_group.attrs['optimizer'] = optimizer
                    
                start_time = time() # To time how long execution takes
                
                avg_iterations = [] # Stores iterations required to reach the precisions in prec_list for each seed, to average over later
                first_loop = True
                epsilon = eps_file["{}/n = {}/{}/Best epsilon".format("Random Ising model", n, optimizer)][()] # Using the optimal step size for the random Ising model (for now)
                
                for seed in seed_list:
                    # Train the QBM & store how many iterations it takes to reach the precisions in prec_list
                    np.random.seed(seed)
                    My_QBM.learn(optimizer=optimizer, q=q, alpha0=alpha0, kmin=kmin, max_qbm_it=max_qbm_it, precision=prec_list, epsilon=epsilon, track_all=False)
                    avg_iterations.append(My_QBM.prec_QBM_track)
                    # Also store the loss per iteration for one seed (for plotting purposes)
                    if first_loop: 
                        # Check if dataset already exists; if so, delete it. Regardless, store the new one.
                        try:
                            loss = opt_group.create_dataset('Model loss', data=My_QBM.loss_QBM_track)
                            grad = opt_group.create_dataset('Average absolute gradient', data=My_QBM.grad_QBM_track)
                        except:
                            del opt_group['Model loss']
                            del opt_group['Average absolute gradient']
                            loss = opt_group.create_dataset('Model loss', data=My_QBM.loss_QBM_track)
                            grad = opt_group.create_dataset('Average absolute gradient', data=My_QBM.grad_QBM_track)
                        first_loop = False
                        
                # Average the number of iterations to reach prec_list over different initializations & store it in the data file
                print(np.array(avg_iterations).shape)
                iters = opt_group.create_dataset('Total iterations to reach prec_list precisions', data=np.average(avg_iterations, axis=0))
                end_time = time()
                print("\t\t Finished on " + datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(end_time - start_time) + " seconds")

    total_time_end = time()
    print("---------- Tests finished on " + datetime.fromtimestamp(total_time_end).strftime("%d-%m-%Y %H:%M:%S") + " in " + str(total_time_end - total_time_start) + " seconds ----------")

# Even if testing is interrupted, make sure the files are properly closed
except KeyboardInterrupt:
    print("Interrupted by user!")
finally:
    f.close()
    eps_file.close()