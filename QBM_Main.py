### MAIN QBM FILE ###
# Contains QBM class & auxiliary functions

import numpy as np
from scipy.linalg import expm, norm

#%%% AUXILIARY FUNCTIONS %%%
def kron_array(a):
    '''Returns the Kronecker product of the elements of array a.'''
    result = a[0]
    for i in range(1,len(a)):
        result = np.kron(result, a[i])
    return result

def weight_mul(b, op_array):
    '''Returns the sum of the operators in op_array weighted by the weights b.
    
    Parameters
    ----------
    b: array of floats, shape (n)
    op_array: array of matrices, shape (n, 2^n, 2^n)
    '''
    return np.tensordot(b, op_array, axes=1)

I = [[1,0],[0,1]]
Z = [[1,0],[0,-1]]
X = [[0,1],[1,0]]
#Y = [[0,-j],[j,0]]

def make_op_arrays(n):
    '''
    Returns a tuple of 3 arrays:
    - op_ar_z: sigma_z operators (size n array of 2^n x 2^n matrices)
    - op_ar_x: sigma_x operators (size n array of 2^n x 2^n matrices)
    - op_ar_zz: pairwise products of sigma_z operators (size n x n array of 2^n x 2^n matrices)
    '''
    Identity_ar = np.array([I for j in range(n)])
    op_list_z = [Identity_ar.copy() for i in range(n)]
    op_list_x = [Identity_ar.copy() for i in range(n)]
    
    for i in range(n):
        op_list_z[i][i] = Z
        op_list_x[i][i] = X

        op_list_z[i] = kron_array(op_list_z[i])
        op_list_x[i] = kron_array(op_list_x[i])

    op_ar_z = np.array(op_list_z)
    op_ar_x = np.array(op_list_x)
    op_ar_zz = np.array([op_ar_z[i] @ op_ar_z[j] for j in range(n) for i in range(n)])
    
    return op_ar_z, op_ar_x, op_ar_zz

def create_H(n, uniform_weights, mag_ratio=1):
    '''Creates a random target Hamiltonian for the QBM to train on.
    The Hamiltonian will be of the form H = sum J_ij z_i z_j + sum h_i x_i.
    
    Parameters
    ----------
    n : positive int, required
        The number of qubits in the model (the sums above range over 1 <= i,j <= n).
    uniform_weights: boolean, required
        If true, weights will be uniform (J_ij = J and h_i = h for all i,j); else, weights will be chosen randomly from [-1,1].
    mag_ratio : float, optional
        Only used if uniform_weights = True; the ratio between J and h.
        
    Returns
    ----------
    A tuple of the following:
    - The Hamiltonian itself (2^n x 2^n matrix)
    - Weights of z-spin interactions J_ij (1D array of size n^2)
    - Weights of transversal field h_i (1D array of size n)
    '''
    _, op_ar_x, op_ar_zz = make_op_arrays(n)
    if(uniform_weights):
        w = np.random.uniform(-1,1)
        b = w/mag_ratio
        J = np.array([w for i in range(n**2)])
        h = np.array([b for i in range(n)])
    else:
        J = np.random.uniform(-1,1,n**2)
        h = np.random.uniform(-1,1,n)
    return (weight_mul(J, op_ar_zz) + weight_mul(h, op_ar_x), J, h)

#%%% QBM CLASS %%%
class QBM():
    '''
    Class that implements a Quantum Boltzmann Machine.
    
    Parameters
    ----------
    eta : 2^n x 2^n complex matrix, required
        Density matrix of the data, used as target for the QBM.
    stat : float, required
        Statistic (usually the expectation value) of eta (CURRENTLY UNUSED, but calculated from eta)
    n : positive int, optional
        The number of qubits (default 2)
    beta: positive float, optional
        Inverse Boltzmann temperature (default 1). A higher beta makes the QBM prioritize pure states more.
    '''
    
    def __init__(self, eta, n=2, beta=1):
        self.n = n
        self.eta = eta
        self.beta = beta
        #self._target_stat = stat
        
        self._op_ar_z, self._op_ar_x, self._op_ar_zz = make_op_arrays(n)
    
    ### EXTERNAL (PUBLIC) METHODS ###
    def learn(self, optimizer, uniform=False, q=1e-3, alpha0=np.sqrt(1e-3), kmin=3, max_qbm_it=50000, precision=1e-7, epsilon=0.25, scale=0.5, track_all=True):
        '''
        Function for learning the QBM, based on the Ising model: H = sum J[i,j] sigma_z[i] sigma_z[j] + sum h[i] sigma_x[i]

        Parameters
        ----------
        optimizer : positive int (in Optimizer dict), required
            The optimizer used for learning the QBM. Permitted values:
            - 'GD': Standard Gradient Descent
            - 'Nesterov_Book': Original Nesterov scheme as proposed by Nesterov in his book on convex optimization
            - 'Nesterov_SBC': Original Nesterov scheme with modified momentum parameter, taken from the paper by Su, Boyd and Canes
            - 'Nesterov_GR': Gradient Restarting Nesterov scheme
            - 'Nesterov_SR': Speed Restarting Nesterov scheme
        uniform : boolean, optional
            Whether the model weights should be uniform (J[i,j] = J and h[i] = h for all i,j) or random (default False, meaning random weights).
        kmin : positive int, optional
            Used only if optimizer == Nesterov_SR; the minimum number of iterations between restarts.
        q : positive float, optional
            Used only if optimizer == Nesterov_Book; q = mu/L is the ratio between the function's strong convexity constant and its Lipschitz constant.
            Since this is difficult to compute in general, it is treated as a hyperparameter here.
        alpha0 : positive float in (0,1), optional
            Used only if optimizer == Nesterov_Book; initialization for the alpha parameter
        max_qbm_it : positive int, optional
            Max number of training iterations of the QBM (default 1000).
        precision: positive float, optional
            The desired precision of the QBM's approximation of the data (default 1e-13).
        epsilon : positive float, optional
            Step size of gradient descent (default 0.05)
        scale : positive float, optional
            Standard deviation of the weight intialization (default 0.5)
        track_all: boolean, optional
            Whether to track all statistics of the QBM (default) or just the loss and total iterations (faster)
            
        Returns
        ----------
        The learned weights, as a dictionary of the form 
        {'J','h'}
        '''
        self._epsilon = epsilon
        self._precision = precision
        
        # Initialize weights randomly (maybe implement some smart start later)
        # UNFINISHED: Start with uniform weights
        # I've yet to figure out if we want to change the learning process if the model is uniform...
        # Realistically, we should, since learning n + n^2 different weights when there are only 2 is excessive,
        # even if all weights converge to the right uniform weights (h[i]_model -> h_exact and likewise for J).
        #if(uniform):
        #    h_unif = np.random.normal(loc=0, scale=scale, size=1)
        #    h = np.array([h_unif for i in range(self.n)])
        #    J_unif = np.random.normal(loc=0, scale=scale, size=1)
        #    J = np.array([[J_unif for i in range(self.n)] for j in range(self.n)])
        #else:
        h = np.random.normal(loc=0, scale=scale, size=self.n)    
        J = np.random.normal(loc=0, scale=scale, size=(self.n, self.n))
        
        # To make the model symmetric & have diagonal 0
        J = (J + J.T)/2
        self._make_diagonal_zero(J)
        
        # Auxiliary variables for momentum + stopping criterion
        h_prev = h.copy()
        J_prev = J.copy()
        
        # Auxiliary variables for the Nesterov schemes
        if(optimizer != 'GD'):            
            h_nest = h.copy()
            J_nest = J.copy()
        
        # Auxiliary variables for Nesterov book scheme:
        if(optimizer == 'Nesterov_Book'):
            alpha = alpha0
            alpha_prev = alpha0
        
        # Compute initial Hamiltonian
        self._H_m_matrix = self._give_H_m_matrix(J, h)
        self._rho_m = expm(-self.beta*self._H_m_matrix)
        self._Z_m = self._rho_m.trace()
        self._rho_m = self._rho_m/self._Z_m
        
        # Initialize trackers
        self._loss_QBM_track = []
        if(track_all):
            self._dl_QBM_track = []
            self._J_QBM_track = []
            self._h_QBM_track = []
            self._stat_QBM_track = []
        
        self.qbm_it = 0
        it_since_restart = 0 # Iterations since the last restart (in case of restarting Nesterov)
        
        # For stopping criterion & speed restarting scheme: 
        # initialize steps (new = ||x_k - x_k-1|| and old = ||x_k-1 - x_k-2||)
        step_old = 0
        step_new = 0
        
        self._target_stat = {'sigma_zz':[], 'sigma_x':[]}
        self._model_stat = {'sigma_zz':[], 'sigma_x':[]}
        
        self._target_stat['sigma_zz'] = [self._give_expectation(self.eta, self._op_ar_zz[i]) for i in range(self.n**2)]
        self._target_stat['sigma_x'] = [self._give_expectation(self.eta, self._op_ar_x[i]) for i in range(self.n)]
        
        # Perform optimization using GD or Nesterov to learn the parameters
        while(self.qbm_it < max_qbm_it):
            self.qbm_it += 1
            it_since_restart += 1
            loss_i = self._give_loss_qbm()
        
            # Compute gradient
            # This can be done exactly for small systems (calculating the expectation value exactly)
            #start_time = time() # DEBUG
            
            self._model_stat['sigma_zz'] = [self._give_expectation(self._rho_m, self._op_ar_zz[i]) for i in range(self.n**2)]
            self._model_stat['sigma_x'] = [self._give_expectation(self._rho_m, self._op_ar_x[i]) for i in range(self.n)]
            
            #end_time = time() # DEBUG
            #print("Gradient computation time: " + str(end_time - start_time)) # DEBUG
            
            # Perform a gradient descent step
            # In the exact case:
            #   dh[i] = eta_expectation(self._op_ar_x[i]) - rho_expectation(self._op_ar_x[i])
            #   dJ[i,j] = eta_expectation(self._op_ar_zz[i,j]) - rho_expectation(self._op_ar_zz[i,j])
            # where eta_expectation(A) = Tr(eta @ A) and rho_expectation(A) = Tr(rho @ A)
            
            dJ = np.array([self._target_stat['sigma_zz'][i] - self._model_stat['sigma_zz'][i] for i in range(self.n**2)]).reshape((self.n, self.n))
            dh = np.array([self._target_stat['sigma_x'][i] - self._model_stat['sigma_x'][i] for i in range(self.n)])

            ##### METHOD: Standard GD #####
            #start_time = time() # DEBUG
            if(optimizer == 'GD'):
                h += -epsilon*dh
                J += -epsilon*dJ
                
                # Make diagonal of J equal to 0 (no self-coupling)
                self._make_diagonal_zero(J)
                
                # Recompute the Hamiltonian with the current parameters
                self._H_m_matrix = self._give_H_m_matrix(J, h)
            
            ##### METHOD: Nesterov #####
            else:
                if(optimizer == 'Nesterov_Book'):
                    # Momentum parameter for Nesterov's original method
                    alpha = (q - alpha**2 + np.sqrt((alpha**2 - q)**2 + 4*alpha**2))/2
                    mom_coef = alpha_prev*(1 - alpha_prev)/(alpha_prev**2 + alpha)
                    alpha_prev = alpha
                else:
                    # Momentum parameter for Su, Boyd and Canes' method (and restarting variants)
                    mom_coef = (self.qbm_it - 1)/(self.qbm_it + 2)
                
                # Update parameters - the d_ (gradient) steps are calculated w.r.t. the auxiliary (_nest) parameters
                h = h_nest - epsilon*dh
                J = J_nest - epsilon*dJ
                
                # Make diagonal of J equal to 0 (no self-coupling)
                self._make_diagonal_zero(J)
                
                # Update Nesterov auxiliary parameters
                h_nest = h + mom_coef*(h - h_prev)
                J_nest = J + mom_coef*(J - J_prev)
            
                # Recompute the Hamiltonian with the *Nesterov* parameters
                self._H_m_matrix = self._give_H_m_matrix(J_nest, h_nest)
            
            ##### COMMON #####
            # For stopping criterion & speed restarting scheme: calculate length of step
            step_new = self._give_distance(J, h, J_prev, h_prev)
            
            # Recompute the density matrix
            self._rho_m = expm(-self.beta*self._H_m_matrix)
            self._Z_m = self._rho_m.trace()
            self._rho_m = self._rho_m/self._Z_m
            
            #end_time = time() # DEBUG
            #print("Step updating time: " + str(end_time - start_time)) # DEBUG
            
            # Calculate post-step loss
            loss_f = self._give_loss_qbm()
            dl = loss_f - loss_i  # How much the loss has been reduced by this GD step
            
            # Add to trackers
            self._loss_QBM_track.append(loss_f)
            if(track_all):
                self._h_QBM_track.append(h.copy())
                self._J_QBM_track.append(J.copy())
                self.stat_QBM_track.append(self._model_stat.copy())         
                self._dl_QBM_track.append(dl)
            
            ##### RESTART NESTEROV #####
            if((optimizer == 'Nesterov_GR' and self._give_x(dJ, dh).T @ (self._give_x(J,h) - self._give_x(J_prev,h_prev)) > 0) or
               (optimizer == 'Nesterov_SR' and it_since_restart > kmin and step_new < step_old)):
                h_nest = h.copy()
                J_nest = J.copy()
                
                it_since_restart = 0
            
            # For speed restarting scheme: update previous step
            step_old = step_new
                
            # VARIABLE STEPSIZE (unused)
            #if dl > 0:                          # Update step size based on dl
                #epsilon = epsilon/2             # Reduce step size if loss was increased (we overstepped)
            #else:
                #epsilon = min(1.01*epsilon, 1)  # Increase step size if loss was decreased (we'll try to be a little more agressive)
                
            # Stop learning if the stopping criterion has been reached
            if step_new < precision:
                break
            
            # Update parameters of previous step
            h_prev = h.copy()
            J_prev = J.copy()
            
        # After training, return the parameters
        self._h = h
        self._J = J
        
        return {'J': self._J, 'h': self._h}
        

    ### INTERNAL (PRIVATE) METHODS ###  
    def _give_H_m_op(self, J, h):
        '''Returns the model Ising Hamiltonian for weights J, h in operator form'''
        # The Hamiltonian is currently hardcoded to be of the following form:
        # H = sum J_ij z_i z_j + sum h_i x_i
        return weight_mul(J.reshape(-1), self._op_ar_zz) + weight_mul(h, self._op_ar_x)
    
    def _give_H_m_matrix(self, J, h):
        '''Returns the model Ising Hamiltonian for weights J, h in (real) matrix form'''
        return self._give_H_m_op(J, h).real
    
    def _give_expectation(self, dens_matrix, operator):
        '''Returns the expectation value of operator (2^n x 2^n matrix) given a density matrix (2^n x 2^n)'''
        return (dens_matrix @ operator).trace()
    
    def _give_distance(self, J1, h1, J2, h2, norm_ord = None):
        '''Returns the distance between the parameter vectors (J1, h1) and (J2, h2) in the given norm (default: Euclidean norm)'''
        return norm(self._give_x(J1.reshape(-1), h1) - self._give_x(J2.reshape(-1), h2), ord=norm_ord)
    def _give_expeta_H_m(self):
        '''Returns Tr(eta * H_m)'''
        #? Wait, what does the 'exp' in the name mean? Expectation value?
        return (self.eta @ self._H_m_matrix).trace()
    
    def _give_Z_m(self):
        '''Returns the model Hamiltonian's partition function'''
        return self._Z
    
    def _give_x(self, J, h):
        '''Returns the model weights stacked as a column vector'''
        return np.concatenate((J.reshape(-1), h))
    
    def _give_L(self):
        '''Returns Tr(eta * log(rho)), the quantum likelihood of the QBM'''
        return -self.beta * self._give_expeta_H_m() - np.log(self._Z_m)
    
    def _give_loss_qbm(self):
        '''Returns the loss of QBM training (here the negative log likelihood is used as loss measure)'''
        return -self._give_L()
    
    def _give_stat_QBM(self):
        '''Returns the expectation values of the spin operators wrt the model'''
        return self._model_stat
    
    def _make_diagonal_zero(self, J):
        '''Replaces all diagonal elements of J with 0.'''
        for i in range(self.n):
            J[i,i] = 0
    
    ### PROPERTIES ###
    @property
    def stat_QBM_track(self):
        return self._stat_QBM_track
    @property
    def dl_QBM_track(self):
        return self._dl_QBM_track
    @property
    def loss_QBM_track(self):
        return self._loss_QBM_track
    @property
    def J_QBM_track(self):
        return self._J_QBM_track
    @property
    def h_QBM_track(self):
        return self._h_QBM_track
    
    #  nontracked properties 
    @property 
    def learned_params(self):
        return {'J':self._J, 'h':self._h}
    @property
    def L(self):
        return self._give_L()
    @property
    def model_stat(self):
        '''Returns the stat for <sigma_z[i] sigma_z[j]> and <sigma_x[i]> after learning finishes'''
        return self._model_stat
