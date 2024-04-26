### SALAMANDER DATA FILE ###
# Contains functions to select and read data from the Salamander Retina dataset.
# MAKE SURE YOU HAVE bint.txt IN THE DATA FOLDER!

import numpy as np

def give_rand_salamander_data(n_neuron = 10):
    '''
    Read data from the Salamander Retina dataset.
    This dataset contains measurements of 160 neurons from a salamander's retina as it watched a movie. The data comes from 297 repeat experiments, each with 953 time points.
    The data is encoded as a list of 283041 (= 297 x 953) binary words of 160 bits each. This function randomly selects n_neuron neurons and one experiment from this set.

    Parameters
    ----------
    n_neuron : int, optional
        The number of neurons to select data for (default 10)
    
    Returns
    ----------
    A dictionary of 4 objects:
    'selected_data' : 953 x n_neuron binary array.
        The data of the n_neuron randomly selected neurons at 953 time points, chosen from one random experiment.
    'experiment_number' : int
        The number of the randomly selected experiment
    'selected_neurons' : 1D array of n_neuron ints
        The numbers of the randomly selected neurons
    'all_data' : 283041 x n_neuron binary array.
        The data of the n_neuron randomly selected neurons across all experiments. Purely for show.
    '''
    n_rows = 160
    n_columns = 283041
    n_experiments = 297
    n_tpoints_per_experiment = 953
    
    selected_neurons = np.random.choice(n_rows, size = n_neuron,replace=False)
    selected_neurons = np.sort(selected_neurons)
    experiment_number = np.random.choice(n_experiments)

    selected_data = np.zeros((n_neuron, n_tpoints_per_experiment))
    all_data = np.zeros((n_neuron, n_columns)) # this is only for showing, not needed

    j = 0
    with open('./Data/bint.txt','rt') as file:
        for i, line in enumerate(file.readlines()):
            if i in selected_neurons:
                selected_data[j] = line.split()[experiment_number*n_tpoints_per_experiment:experiment_number*n_tpoints_per_experiment + n_tpoints_per_experiment]
                all_data[j] = line.split()
                j+=1
    return {'selected_data': (2.*selected_data-1).T, 'experiment_number': experiment_number, 'selected_neurons': selected_neurons, 'all_data': (2.*all_data-1).T}

# Example
#dic = give_rand_salamander_data(20)
#print(dic['selected_data'].shape)
#print(dic['experiment_number'])
#print(dic['selected_neurons'])
#print(dic['all_data'].shape)