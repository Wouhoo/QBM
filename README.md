# QBM
Implementation of a Quantum Boltzmann Machine in Python, using only elementary python libraries like numpy (at least for now).

Started in 2024 for a bachelor thesis project to test the effect of various optimizers (based on Nesterov's Accelerated Gradient method) on the learning speed. Currently still in development.

Files in this repository:
- <b>QBM_Main.py</b>: Contains the QBM class, as well as some auxiliary functions and constants commonly used by other scripts. MAKE SURE YOU HAVE THE OTHER SCRIPTS IN THE SAME FOLDER AS THIS ONE.
- <b>QBM_Example.py</b>: An example of how the QBM may be executed, and how its learning process may be visualised.
- <b>QBM_Tests.py</b>: Code for running tests with the QBM, to see how quickly it learns with different (hyper)parameter settings.
- <b>QBM_Plot.py</b>: Code for plotting the results of the above tests in various ways.
- <b>QBM_TuneParams.py</b>: Code for tuning hyperparameters of the QBM, particularly stepsize.
- <b>QBM_MergeData.py</b>: When running tests in multiple runs and/or on different devices, this code can be used to merge the resulting data files together.
- <b>QBM_Misc.py</b>: Various old code snippets for reference. Not documented very well.