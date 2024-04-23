### CALCULATE CONVERGENCE RATES ###
# Script to calculate convergence rates of various optimizers for a given Hamiltonian.
# According to theory, the number of iterations required to reach precision eps is proportional to log(eps)/log(R), with R the convergence rate;
# by plotting the number of iterations vs eps, we can use a linear fit to find 1/log(R), and derive R from there.

import numpy as np
import matplotlib.pyplot as plt
import h5py     # To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web
from sklearn.linear_model import LinearRegression # For linear fitting
from QBM_Main import filedir

# Parameters for which to compare the optimizers (EDIT HERE)
optimizer_list = ['GD', 'Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR', 'Adam'] # Optimizers to compare
prec_list = [1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]                                 # Precisions for which to compare the optimizers
model = "Random Ising model"                                                      # Model to do the comparison for
n = 8                                                                             # Amount of qubits to do the comparison for
H_number = 3                                                                           # Hamiltonian to do the comparison for (this corresponds to the J/h ratio in case of the Uniform Ising model)

# Data file
f = h5py.File(filedir + '/Data_iters_vs_precision_random.hdf5', 'r')


def calc_rates(file, optimizer_list, prec_list, model, n, H_number, plot=False):
    '''
    Function to calculate convergence rates of optimizers and speedup ratios.

    Parameters
    ----------
    file : HDF5 file, required
        File containing data; for each optimizer in optimizer_list, this must contain a 1D array showing how many iterations the optimzer took to reach each precision in prec_list.
    optimizer_list : 1D array of strings, required
        List of optimizers to compare.
    prec_list : 1D array of positive floats, required
        List of precisions to use in fitting to find the rate.
    model : string, required
        Which model to compare the optimizers for.
    n : positive int, required
        Which qubit amount to compare the optimizers for.
    H_number : positive int, required
        Which Hamiltonian to compare the optimizers for (corresponds to J/h ratio in case of Uniform Ising model).
    plot : boolean, optional
        Whether to plot the results as well (default False)

    Returns
    ----------
    A dictionary of the form {optimizer : rate} for each optimizer in optimizer_list.
    '''

    rates = {}

    x = np.log10(prec_list).reshape(-1,1)   # log(precision)
    for optimizer in optimizer_list:
        y = file['{}/n = {}/Hamiltonian {}/{}/Total iterations to reach prec_list precisions'.format(model, n, H_number, optimizer)][()] # iterations to reach precision

        fit = LinearRegression(fit_intercept = True).fit(x, y) # Choose where if an intercept should be fitted or not
        log_rate = fit.coef_[0] # slope of linear fit = 1/log(R)
        rate = 10**(1/log_rate) # = R

        rates[optimizer] = rate

        if plot:
            plt.scatter(x, y, marker='o', label=optimizer)
            plt.plot(x, fit.predict(x), linestyle='--')

    if plot:
        plt.xlabel("log$_{10}$(Precision)")
        plt.ylabel("No. of iterations")
        plt.legend()      
        plt.show()

    return rates

def calc_speedups(rates):
    '''
    Calculates speedup ratios between optimizers.

    Parameters
    ----------
    rates : dict of the form {optimizer : rate}, required
        Dictionary of optimizers and corresponding convergence rates (as created by the calc_rates function)

    Returns
    ----------
    A 2D array where entry (i,j) is the speedup ratio R_j/R_i, with R_i the ratio of optimizer i (the key of the i-th entry in rates).
    The lower entry (i,j) is, the faster optimizer j is compared to optimizer i (lower rate = better).
    For the fastest optimizer (and *only* the fastest optimizer), all entries in its row are >= 1.
    '''

    rates = list(rates.values())
    no_opts = len(rates)
    return np.array([[rates[j]/rates[i] for j in range(no_opts)] for i in range(no_opts)])
   
    
rates = calc_rates(f, optimizer_list, prec_list, model, n, H_number, plot=True)
print(rates)
#speedups = calc_speedups(rates)
#print(speedups)
print("Best optimizer for this H: ")
print(min(rates, key=rates.get))