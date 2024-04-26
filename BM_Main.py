### MAIN BM FILE ###
# Contains Classical BM class & auxiliary functions

import numpy as np
from numpy import exp, log
#import scipy as sp
#import itertools
#import time
import matplotlib.pyplot as plt

from time import time
from scipy.optimize import fsolve

#%%% AUXILIARY FUNCTIONS %%%
def initialize_data(n, P):
    '''Generates a random dataset with n neurons and P samples.'''
    return np.random.choice([-1.,1.],size=(P,n))

def give_H(h, J, data):
    '''Returns a 1D array of length P (=no. of samples in data), containing the Hamiltonian for each sample (with weights h and J)'''
    n = J.shape[0]
    #print((data@((1-np.eye(n))*J)))
    H_int = np.sum(np.multiply((data@((1-np.eye(n))*J)), data), axis = 1)/2
    #print(H_int)
    H_loc = np.sum(data@h, axis = 1)
    return -1.*(H_int + H_loc)

def give_BM_Pstar(h, J, data):
    '''Returns the non-normalized probability distribution p* for a BM with weights h and J for the data'''
    return np.exp(-give_H(h, J, data))

def give_BM_P(h, J, data):
    '''Returns the normalized probability p for a BM with weights h and J for the data'''
    Pstar = give_BM_Pstar(h, J, data)
    Z = np.sum(Pstar)
    P = Pstar/Z
    return P, Z

def give_log_likelihood_data(h, J, data):
    '''Returns the log-probability for a BM with weights h and J for the data'''
    p ,_ = give_BM_P(h, J, data)
    return np.log(p)

def give_log_likelihood(h, J, data):
    '''Returns the log-likelihood for a BM with weights h and J for the data'''
    out = give_log_likelihood_data(h, J, data)
    return np.sum(out)/data.shape[0]


#%%% BM CLASS %%%
class BM():
    '''
    Class that implements a Classical Boltzmann Machine.
    
    Initialization parameters
    ----------
    data: P x n matrix with binary entries, required
        Each row of the data matrix is one sample of n neurons, i.e. a binary word of length n.
        The goal of the BM is to learn the (underlying) distribution of these binary words.
    '''
        
    def __init__(self, data):
        self._data = data
        self._n = data.shape[1]
        self._P = data.shape[0]
        # all permutations
        self._all_permutations = self._give_all_permutations(self._n)
        self._all_correlations  = np.einsum('ki, kj-> kij', self._all_permutations, self._all_permutations)
        
    def learn(self, alpha=0.99, initialize_option='gaussian', max_bm_it=10000, precision=1e-13, epsilon=0.01, scale=1, method='exact'):
        '''
        Function for learning the BM, based on the model H = sum J[i,j] s[i] s[j] + sum h[i] s[i]

        Parameters
        ----------
        #*** TODO: Add optimizers
        eta : (small) positive float, optional
            Step size of the learning algorithm (default 0.01).
        max_bm_it : int, optional
            Max number of training iterations (default 1000).
        scale : positive float, optional
            Standard deviation of the weight initialization (default 1)
        alpha : float in (0,1), optional
            The momentum coefficient, i.e. how much of the previous weight update is added to the current one (default 0.99)
        initialize_option : 'gaussian', 'uniform', or 'costum', optional. #*** TODO: Take this out?
            The distribution used to intialize weights (default gaussian)
        precision : (small) positive float, optional
            The desired precision of the BM's approximation of the data (default 1e-13)
            The BM will stop learning if all of its weight updates are less than this number (in absolute value).
        method : string, optional #*** TODO: Take this out?
            How to solve for the gradients. Currently, the only option is 'exact'.
            
        Returns
        ----------
        The learned weights, as a dictionary of the form 
        {'J','h'}
        '''
        self.convergence = False
        self.precision = precision

        h, J = self._initialize_BM(self._n, option=initialize_option)
        self._clamped_stat = self._give_clamped_stat()
        # Momentum, to boost the algorithm
        dh_0 = 0
        dJ_0 = 0
    
        self._log_likelihood_track = []
        self._dh_max_track = []
        self._dJ_max_track = []
        self._max_update_track = []
        
        #Z = give_Z(h, J, self._all_permutations)
        
        self._log_likelihood_track.append(give_log_likelihood(h, J, self._data))
        #print(self._give_log_likelihood(h, J, Z))
        self._bm_it = 0  
        while(self._bm_it<max_bm_it):
            self._model_stat, Z = self._give_model_stat(h, J)
            dh = (self._clamped_stat['sigma'] - self._model_stat['sigma'])
            dJ = (self._clamped_stat['sigma_sigma'] - self._model_stat['sigma_sigma'])
            
            #print(dh)
            #print(dJ)
            h += epsilon*(dh.T + alpha*dh_0)
            J += epsilon*(dJ + alpha*dJ_0)
            
            dh_0 = np.copy(dh.T)
            dJ_0 = np.copy(dJ)
            
            
            self._log_likelihood_track.append(give_log_likelihood(h, J, self._data))
            self._dh_max_track.append(np.max(abs(dh)))
            self._dJ_max_track.append(np.max(abs(dJ)))
            #self._log_likelihood_track.append(self._give_log_likelihood(h, J, Z))
            self._bm_it +=1
            #print('iteration: ', self._bm_it)
            #print('np.max(|dh|): ', np.max(abs(dh)))
            if (self._give_convergence()):
                self.convergence = True
                print('BM reached the criteria')
                break
        
        self.h = h
        self.J = J
        return {'J':J, 'h':h}       
     
    ### INTERNAL METHODS ###
    def _initialize_BM(self, n, option='gaussian', scale=1):
        '''Initializes the BM with random weights chosen the given probability distribution (option)'''
        if option=='gaussian':
            # Initial parameters of BM
            h = np.random.normal(loc=0, scale=scale, size=(n, 1))
            J = np.random.normal(loc=0, scale=scale, size=(n, n))
            J = J + J.T
            J = J*(1-np.eye(n))

        elif option== 'uniform':
            h = np.random.uniform(low = -1, high =1, size=(n,1))/n
            J = np.random.uniform(low = -1, high = 1, size=(n,n))
            J = (J+J.T)/(np.sqrt(n))
            J = J*(1-np.eye(n))
        
        elif option== 'costum':
            h = np.ones((n,1))*1.0
            J = 1.0*np.ones((n,n))
            J = J*(1-np.eye(n))
                
        return h, J
    
    def _give_all_permutations(self, n):
        '''Returns all n-bit binary words as a 2^n x n array of floats.'''
        #all_perms = 1.*np.ones((2**n,n))
        all_perms = 1.*np.ones((2**n,n), dtype='float64')
        ran = np.arange(2**n)
        for i in range(n):
            #print(ran%(2**(i+1)))
            all_perms[ran%(2**(i+1))< 2**i,i] = -1.
        return all_perms.astype('float64')

    def _give_clamped_stat(self):
        '''Returns the clamped expectation value of the data (also known as target stat), as a dictionary of the form {'sigma','sigma_sigma'}.'''
        if not hasattr(self, '_clamped_stat'):
            sigma = self._data.mean(axis=0,keepdims = True)
            sigma_sigma = np.einsum('ki,kj->kij',self._data,self._data).mean(axis=0)
    
            self._clamped_stat = {'sigma':sigma, 'sigma_sigma':sigma_sigma}
        return self._clamped_stat
    
    def _give_model_stat(self, h, J, method='exact'):
        '''Returns the model expectation value of the BM, as a dictionary of the form {'sigma','sigma_sigma'}.'''
        p,Z = give_BM_P(h, J, self._all_permutations)
        sigma = np.einsum('k,ki->i',p,self._all_permutations)
        sigma_sigma = np.einsum('k,kij->ij', p, self._all_correlations)
    #    elif method == 'MH':
    #        if not hasattr(self, 's0'):
    #            self.s0 = np.random.choice([-1.,1.],size = (1,J.shape[0]))
    #            #print(self.s0)
    #        samples = give_MH_samples(h, J, option['n_samples'], self.s0)
    #        #print('samples done')
    #        sigma = np.einsum('ki->i',samples)/option['n_samples']
    #        sigma_sigma = np.einsum('ki,kj->ij',samples,samples)/option['n_samples']
    #        self.s0 = samples[-1][None,:] # u have to increase the dimension by one. 
    #        Z = None
    #    #print('give_model done')
    #    elif method == 'MF_LR':
    #        if option['solve_MF'] == 'fsolve':
    #            sigma= give_sigma_MF_v2(h, J)
    #        elif option['solve_MF'] == 'simple':      
    #            sigma, _ = give_sigma_MF(h, J, error_criteria= option['error_criteria'], alpha_MF=option['alpha_MF'], maxiter=option['maxiter'])
    #        sigma_sigma = give_sigma_sigma_LinearResponse(J, sigma)
    #        Z = None
        return {'sigma':sigma, 'sigma_sigma':sigma_sigma}, Z 
    
    def _give_convergence(self):
        '''Checks whether the BM has converged or not.'''
        self.max_update = max(self._dh_max_track[self._bm_it-1],self._dh_max_track[self._bm_it-1])
        return(self.max_update < self.precision)
    #    elif method == 'MH' or method == 'MF_LR':    
    #        if self._bm_it < self.option['n_grad']:
    #            self.max_update = max(np.sum(self._dh_max_track[0:self._bm_it-1 ]),np.sum(self._dJ_max_track[0:self._bm_it-1 ]))/self._bm_it
    #            outpt = False
    #        else:
    #            self.max_update = max(np.sum(self._dh_max_track[self._bm_it-1 -self.option['n_grad']:self._bm_it-1 ]),np.sum(self._dJ_max_track[self._bm_it-1 -self.option['n_grad']:self._bm_it-1 ]))/self.option['n_grad']
    #            output = (self.max_update <self.precision)
    #            
    #        self._max_update_track.append(self.max_update)
    #        
    #        return output
    @property
    def log_likelihood_track(self):
        '''Tracks log likelihood at each step.'''
        return np.array(self._log_likelihood_track)
    @property
    def dh_max_track(self):
        '''Tracks maximum absolute change in h at each step.'''
        return np.array(self._dh_max_track)
    @property
    def dJ_max_track(self):
        '''Tracks maximum absolute change in J at each step.'''
        return np.array(self._dJ_max_track)
    #@property
    #def max_update_track(self):
    #    
    #    return np.array(self._max_update_track)
        

