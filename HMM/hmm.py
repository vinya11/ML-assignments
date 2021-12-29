from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        # print(self.pi)
        # print(f"A {self.A}")
        # print(f"B {self.B}")
        # print(f" state dict {self.state_dict}")
        # print(f" obs dict {self.obs_dict}")
        

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        
        alpha[:,0] = self.pi*self.B[:,O[0]]
        
        for t in range(1,L):
            pink = np.dot(self.A.T,alpha[:,t-1])
            alpha[:,t] = self.B[:,O[t]] * pink 
        return alpha
        
    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        # print(O)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        beta[:,L-1] = 1
        for t in range(L-2,-1,-1):
            one = self.A
            two = (self.B[:,O[t+1]]*beta[:,t+1])
            
            beta[:,t] = np.dot(one,two)
            
        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        
        prob =0
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        # for t in range(0,Osequence.shape[0]):
        #     prob +=alpha[:,t]*beta[:,t]
        return np.random.choice(np.sum(alpha*beta,axis=0))


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        return self.forward(Osequence)*self.backward(Osequence)/self.sequence_prob(Osequence)


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        O = self.find_item(Osequence)
        prod1 = np.zeros((L-1,S))
        prod2 = np.zeros((L-1,S))
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        for t in range(0,L-1):
            beta_b = self.backward(Osequence)[:,t+1]
            a_b_dot_1 = np.multiply(self.A[0,:].T,self.B[:,O[t+1]])
            # print(a_b_dot_1)
            a_b_dot_2 = np.multiply(self.A[1,:].T,self.B[:,O[t+1]])
            prod1[t] = np.multiply(a_b_dot_1,beta_b.T)
            prod2[t] = np.multiply(a_b_dot_2,beta_b.T)
        alpha1 = np.tile(self.forward(Osequence)[0,:-1],(prod1.shape[1],1)).T
        alpha2 = np.tile(self.forward(Osequence)[1,:-1],(prod1.shape[1],1)).T
        # print(alpha1.shape)
        # print(prod1.shape)
        prob[0,:,:] = (alpha1*prod1).T
        prob[1,:,:] = (alpha2*prod2).T
        # print(prob)
        denom = self.sequence_prob(Osequence)
        return prob/denom


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        S = len(self.pi)
        L = len(Osequence)
        O=self.find_item(Osequence)
        delta = np.zeros((S,L))
        small_delta = np.zeros((S,L))
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        delta[:,0] = self.pi*self.B[:,O[0]]
        for t in range(1,L):
            a_delta = np.multiply(self.A.T, delta[:, t-1])
            delta[:,t] = np.multiply(self.B[:,O[t]],np.max(a_delta, axis=1))
            small_delta[:,t] = np.argmax(a_delta,axis=1)
        key_f = self.find_key(self.state_dict,np.argmax(delta[:,L-1],axis=0))
        # print(key_f)
        path.append(key_f)
        for t in range(1,L):
            path_t = small_delta[self.state_dict[path[t-1]],L-t]
            path.append(self.find_key(self.state_dict,path_t))
        path.reverse()

        
        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
