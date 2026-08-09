import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import itertools

# Use quantum-compatible operations on bit representations of particle states to do k-eigenvalue neutron transport
# If done correctly, these operations can be applied to superpositions of particle states to do Monte Carlo neutron
# transport in superposition.
# Using a function to obtain a single number for the k-eigenvalue from the final fission source, use the quantum acceleration
# method presented in https://arxiv.org/abs/1504.06987 to perform this whole operation in O(1/epsilon) time (hopefully)

class QuantumState:
    def __init__(self, Np, Nd, M): # Np is number of qubits representing position, Nd is number of qubits representing direction, M is number of states in superposition
        self.Np = Np
        self.Nd = Nd
        self.Nrng = Np+Nd # number of qubits representing random number
        self.q_tot = self.Np + self.Nd + self.Nrng

        self.M = M
        self.states = np.array([[0]*N]*M, dtype=int)
        self.weights = np.array([1/math.sqrt(M)]*M, dtype=float)

    # return the entire quantum state in superposition
    def getState(self):
        return np.array([self.weights[i]*self.states[i] for i in range(self.M)])
    
    # print the current quantum state to the console
    def printState(self):
        print("Current Quantum State: ", end='')
        for i in range(self.M):
            print(str(self.weights[i]) + "|" + str(self.states[i]) + ">", end='')
            if i < self.M-1:
                print(" + ", end='')
        print("")
    
    # update the bit representation of a state
    def updateState(self, m, new_state):
        if(len(new_state) != self.N):
            raise Exception("state is not correct size")
        for i in range(self.M):
            if(new_state[i] != 0 and new_state[i] != 1):
                raise Exception("state is not in binary format")
        self.states[m] = new_state
    
    

# 1 quantum state
qs = QuantumState(4,1)
qs.updateState(0,[1,0,1,0]) # bit representation of 10/16 = 0.625
qs.printState()

# 2 states in superposition
'''qs = QuantumState(4,2)
qs.updateState(0,[1,0,1,0]) # bit representation of 10/16 = 0.625
qs.updateState(1,[0,1,1,1]) # bit representation of 7/16 = 0.4375
qs.printState()'''
