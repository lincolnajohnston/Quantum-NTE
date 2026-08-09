import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools

def get_M_1D(N):
    matrix = np.zeros((N, N))
    # all elements scaled by 1/N because the inner products scale with the area under each basis function, which scales with 1/N
    c = 1/(N+1)
    matrix[0:N,0:N] += np.diag(4*c*np.ones(N))
    matrix[0:N,0:N] += np.diag(c*np.ones(N-1), k=1)
    matrix[0:N,0:N] += np.diag(c*np.ones(N-1), k=-1)
    return matrix

def get_P(h, H):
    P = np.zeros((2**h - 1, 2**H - 1))
    dh = h-H
    expand_vec_prefix = np.arange(1/(2**dh), 1, 1/(2**dh), dtype=float)
    expand_vec = np.concatenate((expand_vec_prefix, [1], np.flip(expand_vec_prefix)))
    for i in range(2**H - 1):
        P[i*int(2**dh):(i+2)*int(2**dh)-1,i] = expand_vec
    return P
    

D = 1
H = 2
nH = 2**H - 1
h = 5
nh = 2**h - 1

M_h = get_M_1D(nh) # fine mass matrix
M_H = get_M_1D(nH) # coarse mass matrix

P = get_P(h, H) # assume we can block encode this matrix efficiently
M_H_test = np.transpose(P) @ M_h @ P # this should be equal to M_H

# assume we can apply each of these matrices efficiently
# M is a tridiagonal Toeplitz matrix which can be block encoded with constant alpha and ancillas
# Because M has a constant condition number (highest and lowest eigenvalue scale with h in the same way), then
# QSVT can be used to efficiently each of the square roots and inverses of M_h and M_H in O(polylog(n)) time
M_h_sqrt = np.sqrt(M_h)
M_H_sqrt = np.sqrt(M_H)
M_h_sqrt_inv = np.linalg.inv(M_h_sqrt)
M_H_sqrt_inv = np.linalg.inv(M_H_sqrt)

# coarse amplitude vector:
c_H = np.random.random(nH)
c_H = c_H / np.linalg.norm(c_H)

# mass-weighted coordinates:
chi_H = M_H_sqrt @ c_H
chi_H = chi_H / np.linalg.norm(chi_H)

A = M_h_sqrt @ P @ M_H_sqrt_inv # this should be an isometry that we can apply efficiently?
c_h = M_h_sqrt_inv @ A @ chi_H

print("done")