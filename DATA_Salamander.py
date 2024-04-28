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
    'selected_data' : 953 x n_neuron binary array
        The data of the n_neuron randomly selected neurons at 953 time points, chosen from one random experiment.
    'experiment_number' : int
        The number of the randomly selected experiment.
    'selected_neurons' : 1D array of n_neuron ints
        The numbers of the randomly selected neurons.
    'all_data' : 283041 x n_neuron binary array
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

def give_all_permutations(n):
        '''Returns all n-bit binary words as a 2^n x n array of floats.'''
        #all_perms = 1.*np.ones((2**n,n))
        all_perms = 1.*np.ones((2**n,n), dtype='float64')
        ran = np.arange(2**n)
        for i in range(n):
            #print(ran%(2**(i+1)))
            all_perms[ran%(2**(i+1))< 2**i,i] = -1.
        return all_perms.astype('float64')

def give_salamander_dens_matrix(n = 10):
    '''
    Read data from the Salamander Retina dataset and transform it into a density matrix. 
    For more info on this dataset, see the give_rand_salamander_data function.

    Parameters
    ----------
    n : int, optional
        The number of neurons to select data for (default 10)
    
    Returns
    ----------
    A dictionary of 3 objects:
    'dens_matrix' : 2^n x 2^n density matrix
        The data of the n randomly selected neurons at 953 time points, chosen from one random experiment, transformed into a 2^n x 2^n target density matrix.
    'experiment_number' : int
        The number of the randomly selected experiment.
    'selected_neurons' : 1D array of n ints
        The numbers of the randomly selected neurons.
    '''

    salamander_data = give_rand_salamander_data(n)
    data = salamander_data['selected_data']
    no_samples = data.shape[0]
    perms = give_all_permutations(n)

    q = np.zeros(2**n) # Empirical probability distribution (1D array of size 2^n)
    for i in range(2**n):
        perm = perms[i]
        count = 0
        for j in range(no_samples):
            if np.array_equal(data[j], perm):
                count += 1
        q[i] = count/no_samples

    #print(q)
    #print(q.shape)
    #print(np.sum(q))
    psi = np.sqrt(q).reshape(2**n,1)
    eta = psi @ psi.T # 2^n x 2^n density matrix representing the classical data

    return {'dens_matrix': eta, 'experiment_number': salamander_data['experiment_number'], 'selected_neurons': salamander_data['selected_neurons']}

# Example
#n = 8

#dic = give_rand_salamander_data(n)
#print(dic['selected_data'])
#print(dic['selected_data'].shape)
#print(dic['experiment_number'])
#print(dic['selected_neurons'])
#print(dic['all_data'].shape)

#print('\n')

#dic2 = give_salamander_dens_matrix(n)
#print(dic2['dens_matrix'])
#print(dic2['dens_matrix'].shape)
#print(dic2['experiment_number'])
#print(dic2['selected_neurons'])