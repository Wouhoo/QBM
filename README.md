# QBM
Implementation of a Quantum Boltzmann Machine in Python, using only elementary python libraries like numpy (at least for now).

Started in 2024 for a bachelor thesis project to test the effect of various optimizers (based on Nesterov's Accelerated Gradient method) on the learning speed. Currently still in development.

Files in this repository:
- <b>QBM_Main.py</b>: Contains the QBM class, as well as some auxiliary functions and constants commonly used by other scripts. MAKE SURE YOU HAVE THE OTHER SCRIPTS IN THE SAME FOLDER AS THIS ONE.
- <b>QBM_Example.py</b>: An example of how the QBM may be executed, and how its learning process may be visualised.
- <b>QBM_Tests.py</b>: DEPRECATED - USE v2 INSTEAD! Code for running tests with the QBM, to see how quickly it learns with different (hyper)parameter settings.
- <b>QBM_Tests_v2.py</b>: Code for running tests with the QBM, to see how quickly it learns with different (hyper)parameter settings. Specifically, this tests for various different models, numbers of qubits, and Hamiltonian parameters how many iterations each optimizer takes to reach certain precisions.
- <b>QBM_Salamander.py</b>: Code for testing the QBM's performance when learning the Salamander Retina dataset.
- <b>QBM_Plot.py</b>: Code for plotting the results of the above tests in various ways.
- <b>QBM_TuneParams.py</b>: Code for tuning hyperparameters of the QBM, particularly stepsize.
- <b>QBM_Rates.py</b>: Code for calculating the convergence rates of various optimizers, and their relative speedup compared to eachother.
- <b>QBM_MergeData.py</b>: When running tests in multiple runs and/or on different devices, this code can be used to merge the resulting data files together.
- <b>QBM_Misc.py</b>: Various old code snippets for reference. Not documented very well.

- <b>BM_Main.py</b>: Contains the classical BM class.
- <b>BM_Example.py</b>: An example of how the BM may be executed, and how its learning process may be visualised.

- <b>DATA_Salamander.py</b>: Code for extracting data from the Salamander Retina dataset (Schneidman et al., 2006) and transforming it into the correct format for learning (Q)BM.

- <b>Data folder</b>: Folder containing data files in HDF5 format. The most important files here:
  - <b>Data_iters_vs_precision_random.hdf5</b>: Results of testing using QBM_Tests_v2.py for all optimizers and all n, for the Random Ising model, up to precision 1e-7. Data for the Uniform Ising model is still underway.
  - <b>Eps_Data_short_modified.hdf5</b>: Contains the optimal step sizes for all (model, n, optimizer) combo's.
  - <b>bint.txt</b>: The full Salamander Retina dataset.

- <b>Plots folder</b>: Folder containing an assortment of plots. There's currently 3 types:
  - <b>iters_ plots</b>: Shows how many iterations each optimizer takes on average for different precisions, for a given Hamiltonian with a certain model and number of qubits. This demonstrates how much Nesterov's scheme (particularly the restarting variants) speeds up the learning process compared to gradient descent.
  - <b>example_ plots</b>: These show a single run of the QBM each (for one particular model, precision, and number of qubits, as indicated in the name). Useful to get an idea of how the different optimizers learn.
  - <b>other plots</b>: These show how the various optimizers compare for the same Hamiltonian. Naming convention: \<model>\_\<precision>\_\<number of qubits>\_\<Hamiltonian number>.png

To easily view HDF5 files, try myhdf5.hdfgroup.org in your browser, or use the VSCode extension H5Web.
