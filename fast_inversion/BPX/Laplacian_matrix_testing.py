import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools
import FEM_BPX_helpers as FEM

# return a FEM mass matrix where each term is weighted by a piecewise-constant value (like the absorption cross section)
# L is number of levels, consts is an array of constants defined in each cell of the FEM discretization
def get_1D_weighted_mass_matrix(L, consts):
    # make sure consts is the right size
    if len(consts) != int(2**(L)):
        raise ValueError("consts matrix is not the correct size/shape")
    h = 1/(2**L)
    A = np.diag(2*consts[:-1])
    A += np.diag(2*consts[1:])
    A += np.diag(consts[1:-1], k=1)
    A += np.diag(consts[1:-1], k=-1)
    return (h/6)*A # weight the matrix by a scalar

# return a FEM laplacian matrix where each term is weighted by a piecewise-constant value (like the diffusion coefficient)
# L is number of levels, consts is an array of constants defined in each cell of the FEM discretization
def get_1D_weighted_laplacian_matrix(L, consts):
    # make sure consts is the right size
    if len(consts) != int(2**(L)):
        raise ValueError("consts matrix is not the correct size/shape")
    h = 1/(2**L)
    A = np.diag(consts[:-1])
    A += np.diag(consts[1:])
    A += np.diag(-1*consts[1:-1], k=1)
    A += np.diag(-1*consts[1:-1], k=-1)
    return (1/h)*A # weight the matrix by a scalar

sigma_a = 99999
D = 6
L = 4
M = get_1D_weighted_mass_matrix(L,np.ones(int(2**L)))
P = get_1D_weighted_laplacian_matrix(L,np.ones(int(2**L)))

A_3D = np.kron(np.kron(M,M),M)
L_3D = np.kron(np.kron(P,M),M) + np.kron(np.kron(M,P),M) + np.kron(np.kron(M,M),P)

K = L_3D + A_3D
K_inv = np.linalg.inv(K)
K_inv_min = np.min(K_inv) # maximum value of the K_inv matrix, if we can show that the minimum value is always non-negative, that will help us show that the principal eigenvector of the equation is non-negative, which will help us show that the uniform superposition state has constatnt overlap with any solution of it

print("done")