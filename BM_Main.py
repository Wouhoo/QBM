### MAIN BM FILE ###
# Contains Classical BM class & auxiliary functions

import numpy as np
from scipy.linalg import norm
#import itertools
from time import time

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
        
    def learn(self, optimizer, initialize_option='gaussian', max_bm_it=10000, precision=1e-13, epsilon=0.01, scale=1, method='exact', 
                q=1e-3, alpha0=np.sqrt(1e-3), kmin=3, beta1=0.9, beta2=0.999, adam_eps=1e-6):
        '''
        Function for learning the BM, based on the model H = sum J[i,j] s[i] s[j] + sum h[i] s[i]

        General parameters
        ----------
        optimizer : string, required
            The optimizer used for learning the BM. Permitted values:
            - 'GD': Standard Gradient Descent.
            - 'Nesterov_Book': Original Nesterov scheme as proposed by Nesterov in his book on convex optimization (2004, 2nd edition in 2018).
            - 'Nesterov_SBC': Original Nesterov scheme with modified momentum parameter, taken from the paper by Su, Boyd and Candes (2015).
            - 'Nesterov_GR': Gradient Restarting Nesterov scheme, as proposed by O'Donoghue and Candes (2012).
            - 'Nesterov_SR': Speed Restarting Nesterov scheme, as proposed by Su, Boyd and Candes (2015).
            - 'Adam': ADAptive Momentum optimizer proposed by Kingma and Ba (2017)
        eta : (small) positive float, optional
            Step size of the learning algorithm (default 0.01).
        max_bm_it : int, optional
            Max number of training iterations (default 1000).
        scale : positive float, optional
            Standard deviation of the weight initialization (default 1)
        initialize_option : 'gaussian', 'uniform', or 'costum', optional. #*** TODO: Take this out?
            The distribution used to intialize weights (default gaussian)
        precision : (small) positive float, optional
            The desired precision of the BM's approximation of the data (default 1e-13)
            The BM will stop learning if all of its weight updates are less than this number (in absolute value).
        method : string, optional #*** TODO: Take this out?
            How to solve for the gradients. Currently, the only option is 'exact'.

        Parameters for specific optimizers
        ----------
        q : positive float, optional
            Used only if optimizer == Nesterov_Book; used for calculating the momentum parameter.
            q = mu/L is the ratio between the function's strong convexity constant and its Lipschitz constant. 
            Since this is difficult to compute in general, it is treated as a hyperparameter here.
        alpha0 : positive float in (0,1), optional
            Used only if optimizer == Nesterov_Book; initialization for the alpha parameter, used in calculating the momentum parameter.
        kmin : positive int, optional
            Used only if optimizer == Nesterov_SR; the minimum number of iterations between restarts.
        beta1 : positive float in (0,1), optional
            Used only if optimizer == Adam; the exponential decay rate of the first momentum.
        beta2 : positive float in (0,1), optional
            Used only if optimizer == Adam; the exponential decay rate of the second momentum.
        adam_eps : small positive float, optional
            Used only if optimizer == Adam; a small positive constant used to guarantee numerical stability in calculations.
            
        Returns
        ----------
        The learned weights, as a dictionary of the form 
        {'J','h'}
        '''
        self.convergence = False
        self.precision = precision

        # Initialize weights & compute target statistics
        h, J = self._initialize_BM(self._n, option=initialize_option)
        self._clamped_stat = self._give_clamped_stat()

        ### AUXILIARY VARIABLES ###
        # Auxiliary variables for momentum
        h_prev = h.copy()
        J_prev = J.copy()

        # Auxiliary variables for Nesterov schemes
        if(optimizer in ['Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']):            
            h_nest = h.copy()
            J_nest = J.copy()

        # Auxiliary variables for Nesterov book scheme
        if(optimizer == 'Nesterov_Book'):
            alpha = alpha0
            alpha_prev = alpha0

        # Auxiliary variables for Adam scheme
        if(optimizer == 'Adam'):
            mJ = np.zeros((self._n, self._n))
            mJ_hat = np.zeros((self._n, self._n))
            vJ = np.zeros((self._n, self._n))
            vJ_hat = np.zeros((self._n, self._n))
            mh = np.zeros((self._n, 1))
            mh_hat = np.zeros((self._n, 1))
            vh = np.zeros((self._n, 1))
            vh_hat = np.zeros((self._n, 1))
    
        # Initialize trackers
        self._log_likelihood_track = []
        self._dh_max_track = []
        self._dJ_max_track = []
        self._step_track = []
        
        self._log_likelihood_track.append(give_log_likelihood(h, J, self._data))
        self._bm_it = 0  

        it_since_restart = 0
        step_old = 0
        step_new = 0

        # Perform optimization until converged or timed out
        while(self._bm_it<max_bm_it):
            self._bm_it +=1

            # Calculate model statistics
            if(optimizer in ['Nesterov_Book', 'Nesterov_SBC', 'Nesterov_GR', 'Nesterov_SR']):
                self._model_stat, Z = self._give_model_stat(h_nest, J_nest)
            else:
                self._model_stat, Z = self._give_model_stat(h, J)

            # Calculate weight updates
            dh = (self._clamped_stat['sigma'] - self._model_stat['sigma'])
            dJ = (self._clamped_stat['sigma_sigma'] - self._model_stat['sigma_sigma'])
            
            ### METHOD: Standard GD ###
            if(optimizer == 'GD'):
                h += epsilon*dh
                J += epsilon*dJ

            ### METHOD: Adam ###
            elif(optimizer == 'Adam'):
                # Compute first and second momentum
                mJ = beta1 * mJ + (1-beta1) * dJ
                mh = beta1 * mh + (1-beta1) * dh
                vJ = beta2 * vJ + (1-beta2) * dJ**2
                vh = beta2 * vh + (1-beta2) * dh**2

                #print(mJ.shape, mh.shape, vJ.shape, vh.shape, '\n')

                # Correct bias
                mJ_hat = mJ / (1 - beta1**self._bm_it)
                mh_hat = mh / (1 - beta1**self._bm_it)
                vJ_hat = vJ / (1 - beta2**self._bm_it)
                vh_hat = vh / (1 - beta2**self._bm_it)

                #print(mJ_hat.shape, mh_hat.shape, vJ_hat.shape, vh_hat.shape, '\n')

                # Update parameters
                J += epsilon * mJ_hat/(np.sqrt(vJ_hat) + adam_eps)
                h += epsilon * mh_hat/(np.sqrt(vh_hat) + adam_eps)

            ### METHOD: Nesterov ###
            else:
                if(optimizer == 'Nesterov_Book'):
                    # Momentum parameter for Nesterov's original method
                    alpha = (q - alpha**2 + np.sqrt((alpha**2 - q)**2 + 4*alpha**2))/2
                    mom_coef = alpha_prev*(1 - alpha_prev)/(alpha_prev**2 + alpha)
                    alpha_prev = alpha
                else:
                    # Momentum parameter for Su, Boyd and Canes' method (and restarting variants)
                    mom_coef = (self._bm_it - 1)/(self._bm_it + 2)
                
                # Update parameters (notice that the dJ/dh gradient steps are calculated w.r.t. the Hamiltonian evaluated in the auxiliary (_nest) parameters)
                h = h_nest + epsilon*dh
                J = J_nest + epsilon*dJ
                
                # Update Nesterov auxiliary parameters
                h_nest = h + mom_coef*(h - h_prev)
                J_nest = J + mom_coef*(J - J_prev)
            
            ### COMMON ###
            self._log_likelihood_track.append(give_log_likelihood(h, J, self._data))

            step_new = self._give_distance(h, J, h_prev, J_prev)
            self._step_track.append(step_new)

            self._dh_max_track.append(np.max(abs(dh)))
            self._dJ_max_track.append(np.max(abs(dJ)))

            #print('iteration: ', self._bm_it) #DEBUG
            #print('np.max(|dh|): ', np.max(abs(dh))) #DEBUG

            # Check for convergence
            # NOTE: In the original version of the BM, learning stopped as soon as max abs dh got below the required precision (dJ was ignored).
            # In the current version, it is not |dh| which determines this, but |h - h_prev| and |J - J_prev|. If all coordinates of these values are below the precision, learning stops.
            if (step_new < self.precision):
                self.convergence = True
                print('BM reached the criteria')
                break

            ### RESTART NESTEROV ###
            if((optimizer == 'Nesterov_GR' and self._give_x(dh, dJ).T @ (self._give_x(h, J) - self._give_x(h_prev, J_prev)) > 0) or
               (optimizer == 'Nesterov_SR' and it_since_restart > kmin and step_new < step_old)):
                # Kill the momentum
                h_nest = h.copy()
                J_nest = J.copy()
                
                it_since_restart = 0

            # Update parameters of previous step
            h_prev = h.copy()
            J_prev = J.copy()
            step_old = step_new
        
        # Return parameters after training
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
            sigma = (self._data.mean(axis=0,keepdims = True)).T
            sigma_sigma = np.einsum('ki,kj->kij',self._data,self._data).mean(axis=0)
    
            self._clamped_stat = {'sigma':sigma, 'sigma_sigma':sigma_sigma}
        return self._clamped_stat
    
    def _give_model_stat(self, h, J, method='exact'):
        '''Returns the model expectation value of the BM, as a dictionary of the form {'sigma','sigma_sigma'}.'''
        p,Z = give_BM_P(h, J, self._all_permutations)
        sigma = np.einsum('k,ki->i',p,self._all_permutations).reshape(self._n,1)
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
    
    #def _give_convergence(self):
    #    '''Checks whether the BM has converged or not.'''
    #    self.max_update = max(self._dh_max_track[self._bm_it-1],self._dh_max_track[self._bm_it-1])
    #    return(self.max_update < self.precision)
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

    def _give_x(self, h, J):
        '''Returns the weights h and J stacked as a column vector'''
        return np.concatenate((h.reshape(-1), J.reshape(-1)))

    def _give_distance(self, h1, J1, h2, J2, norm_ord = np.inf):
        '''Returns the distance between the parameter vectors (h1, J1) and (h2, J2) in the given norm (default: Maximum norm)'''
        return norm(self._give_x(h1, J1) - self._give_x(h2, J2), ord=norm_ord)

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
    @property
    def step_track(self):
        '''Tracks maximum absolute change in the parameters at each step.
        This is used for the stopping criterion.'''
        return np.array(self._step_track)

    #@property
    #def max_update_track(self):
    #    
    #    return np.array(self._max_update_track)
        

