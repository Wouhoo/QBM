### EXAMPLE BM EXECUTION ###
# Shows an example of how the Classical BM can be executed, and how the learning process can be visualised

import numpy as np
import matplotlib.pyplot as plt
from BM_Main import BM, initialize_data
from DATA_Salamander import give_rand_salamander_data

# Parameters
n = 8
P = 100
max_bm_it = 100000
precision = 1e-3

# Initialize & learn BM
### Random data ###
# data = initialize_data(n,P) 

### Salamander data ###
salamander_data = give_rand_salamander_data(n)
print('Selected experiment: ' + str(salamander_data['experiment_number']))
print('Selected neurons: ' + str(salamander_data['selected_neurons']))
data = salamander_data['selected_data']

MyBM = BM(data)
result = MyBM.learn(max_bm_it=max_bm_it, precision=precision)

# Uncomment to see what the learned weights look like & how long learning took
print("Learned J:")
print(MyBM.J)
print("Learned h:")
print(MyBM.h)
print("Learning stopped after:")
print(str(MyBM._bm_it) + " iterations")

#%%% PLOT %%%

plt.figure(figsize = (16,8))

# Plot parameters (EDIT HERE)
close_up = False    # Whether to show close-ups or the full plot
eps = 0.05          # For close-ups: how far the plot extends from the true value
xlim = MyBM._bm_it  # To limit x axis on the right side

# Plot model loss
plt.subplot(1,2,1)
plt.plot(-MyBM.log_likelihood_track, label="Model loss")
plt.xscale("log")
plt.yscale("log")
plt.title("Evolution of BM loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
if(close_up):
    plt.ylim(-eps, eps) # For a close-up
plt.xlim((0,xlim))
plt.legend()

# Max dh & dJ
plt.subplot(1,2,2)
plt.plot(MyBM.dh_max_track, label='max dh')
plt.plot(MyBM.dJ_max_track, label='max dJ')
plt.axhline(y = MyBM.precision, linestyle='--', label='Required precision')
plt.xscale("log")
plt.yscale("log")
plt.title("Evolution of h")
plt.xlabel("Iteration")
plt.ylabel("h")
plt.xlim((0,xlim))
plt.legend()

plt.show()