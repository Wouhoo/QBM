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
def give_all_permutations(n):
    #all_perms = 1.*np.ones((2**n,n))
    all_perms = 1.*np.ones((2**n,n), dtype='float64')
    ran = np.arange(2**n)
    for i in range(n):
        #print(ran%(2**(i+1)))
        all_perms[ran%(2**(i+1))< 2**i,i] = -1.
    return all_perms.astype('float64')

def give_H(θ, w, data):
    ''' gives 1-dim array of length P, containing H  '''
    n = w.shape[0]
    #print((data@((1-np.eye(n))*w)))
    H_int = np.sum(np.multiply((data@((1-np.eye(n))*w)), data), axis = 1)/2
    #print(H_int)
    H_loc = np.sum(data@θ, axis = 1)
    return -1.*(H_int + H_loc)

def give_BM_Pstar(θ, w, data):
    ''' returns p star, which is not normalized '''
    return np.exp(-give_H(θ, w, data))

def give_BM_P(θ, w, data):
    Pstar = give_BM_Pstar(θ, w, data)
    Z = np.sum(Pstar)
    P = Pstar/Z
    return P, Z

def give_log_likelihood_data(θ, w, data):
    p ,_ = give_BM_P(θ, w, data)
    return np.log(p)

def give_log_likelihood(θ, w, data):
    out = give_log_likelihood_data(θ, w, data)
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
        self._all_permutations = give_all_permutations(self._n)
        self._all_correlations  = np.einsum('ki, kj-> kij', self._all_permutations, self._all_permutations)
        
    def learn(self, η=0.01, max_bm_it=1000, scale=1, alpha=0.99, initialize_option='gaussian', tolerance=1e-13, method='exact'):
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
        tolerance : (small) positive float, optional
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
        self.tolerance = tolerance

        θ, w = self._initialize_BM(self._n, option=initialize_option)
        self._clamped_stat = self._give_clamped_stat()
        # Momentum, to boost the algorithm
        dθ_0 = 0
        dw_0 = 0
    
        self._log_likelihood_track = []
        self._dθ_max_track = []
        self._dw_max_track = []
        self._max_update_track = []
        
        #Z = give_Z(θ, w, self._all_permutations)
        
        self._log_likelihood_track.append(give_log_likelihood(θ, w, self._data))
        #print(self._give_log_likelihood(θ, w, Z))
        self._bm_it = 0  
        while(self._bm_it<max_bm_it):
            self._model_stat, Z = self._give_model_stat(θ, w, method=method)
            dθ = (self._clamped_stat['sigma'] - self._model_stat['sigma'])
            dw = (self._clamped_stat['sigma_sigma'] - self._model_stat['sigma_sigma'])
            
            #print(dθ)
            #print(dw)
            θ += η*(dθ.T + alpha*dθ_0)
            w += η*(dw + alpha*dw_0)
            
            dθ_0 = np.copy(dθ.T)
            dw_0 = np.copy(dw)
            
            
            self._log_likelihood_track.append(give_log_likelihood(θ, w, self._data))
            self._dθ_max_track.append(np.max(abs(dθ)))
            self._dw_max_track.append(np.max(abs(dw)))
            #self._log_likelihood_track.append(self._give_log_likelihood(θ, w, Z))
            self._bm_it +=1
            #print('iteration: ', self._bm_it)
            #print('np.max(|dθ|): ', np.max(abs(dθ)))
            if (self._give_convergence()):
                self.convergence = True
                print('BM reached the criteria')
                break
        
        self.θ = θ
        self.w = w
        return {'θ':θ, 'w':w}       
     
    ### INTERNAL METHODS ###
    def _initialize_BM(n, option='gaussian', scale=1):
        if option=='gaussian':
            # Initial parameters of BM
            θ = np.random.normal(loc=0, scale=scale, size=(n, 1))
            w = np.random.normal(loc=0, scale=scale, size=(n, n))
            w = w + w.T
            w = w*(1-np.eye(n))

        elif option== 'uniform':
            θ = np.random.uniform(low = -1, high =1, size=(n,1))/n
            w = np.random.uniform(low = -1, high = 1, size=(n,n))
            w = (w+w.T)/(np.sqrt(n))
            w = w*(1-np.eye(n))
        
        elif option== 'costum':
            θ = np.ones((n,1))*1.0
            w = 1.0*np.ones((n,n))
            w = w*(1-np.eye(n))
                
        return θ, w

    def _give_clamped_stat(self):
        if not hasattr(self, '_clamped_stat'):
            sigma = self._data.mean(axis=0,keepdims = True)
            sigma_sigma = np.einsum('ki,kj->kij',self._data,self._data).mean(axis=0)
    
            self._clamped_stat = {'sigma':sigma, 'sigma_sigma':sigma_sigma}
        return self._clamped_stat
    
    
    def _give_log_likelihood(self, θ, w, Z):
        L = 0
        for s in self._data:
            ss = np.outer(s,s)
            # L += (1/2* (w.reshape(-1)@(ss.reshape(-1))) + (θ@s)) - log(Z)
            #print('ps-Z: ', 1/2* (w.reshape(-1)@(np.outer(s,s).reshape(-1))) + (θ@s) - log(Z))
            L += log(exp(1/2* (w.reshape(-1)@(np.outer(s,s).reshape(-1))) + (θ@s))/Z)
        return L/self._P    
    
    def _give_model_stat(self, θ, w, method='exact'):
        if method == 'exact':
            p,Z = give_BM_P(θ, w, self._all_permutations)
            sigma = np.einsum('k,ki->i',p,self._all_permutations)
            sigma_sigma = np.einsum('k,kij->ij', p, self._all_correlations)
    #    elif method == 'MH':
    #        if not hasattr(self, 's0'):
    #            self.s0 = np.random.choice([-1.,1.],size = (1,w.shape[0]))
    #            #print(self.s0)
    #        samples = give_MH_samples(θ, w, option['n_samples'], self.s0)
    #        #print('samples done')
    #        sigma = np.einsum('ki->i',samples)/option['n_samples']
    #        sigma_sigma = np.einsum('ki,kj->ij',samples,samples)/option['n_samples']
    #        self.s0 = samples[-1][None,:] # u have to increase the dimension by one. 
    #        Z = None
    #    #print('give_model done')
    #    elif method == 'MF_LR':
    #        if option['solve_MF'] == 'fsolve':
    #            sigma= give_sigma_MF_v2(θ, w)
    #        elif option['solve_MF'] == 'simple':      
    #            sigma, _ = give_sigma_MF(θ, w, error_criteria= option['error_criteria'], alpha_MF=option['alpha_MF'], maxiter=option['maxiter'])
    #        sigma_sigma = give_sigma_sigma_LinearResponse(w, sigma)
    #        Z = None
        return {'sigma':sigma, 'sigma_sigma':sigma_sigma}, Z 
    
    def _give_convergence(self, method='exact'):
        if method == 'exact':
            self.max_update = max(self._dθ_max_track[self._bm_it-1],self._dθ_max_track[self._bm_it-1])
            return( self.max_update<self.tolerance)
    #    elif method == 'MH' or method == 'MF_LR':    
    #        if self._bm_it < self.option['n_grad']:
    #            self.max_update = max(np.sum(self._dθ_max_track[0:self._bm_it-1 ]),np.sum(self._dw_max_track[0:self._bm_it-1 ]))/self._bm_it
    #            outpt = False
    #        else:
    #            self.max_update = max(np.sum(self._dθ_max_track[self._bm_it-1 -self.option['n_grad']:self._bm_it-1 ]),np.sum(self._dw_max_track[self._bm_it-1 -self.option['n_grad']:self._bm_it-1 ]))/self.option['n_grad']
    #            output = (self.max_update <self.tolerance)
    #            
    #        self._max_update_track.append(self.max_update)
    #        
    #        return output
    @property
    def log_likelihood_track(self):
        return np.array(self._log_likelihood_track)
    @property
    def dθ_max_track(self):
        return np.array(self._dθ_max_track)
    @property
    def dw_max_track(self):
        return np.array(self._dw_max_track)
    @property
    def max_update_track(self):
        return np.array(self._max_update_track)
        

