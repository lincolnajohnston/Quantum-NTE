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
import FEM_BPX_helpers as FEM

# trying to visualize the matrices from "Quantum Realization of the Finite Element Method" by Deiml M, Peterseim D and make 
# sure they can be applied effectively and work as preconditioners

# the domain goes from 0 to 1
D = 1
L = 5 # number of levels of BPX preconditioner
sparse = True
n_fine = int(math.pow(2,L)) # number of points in finest level
h_fine = 1/n_fine
CF_col_sections = [(2**l-1)**D for l in range(1,L+1)] # list of number of basis functions in each level

C_L = FEM.getC_l(D, L)
F = FEM.get_F(D, L)

C_F_test = C_L @ F

CF = csr_matrix((D * 2**(D*(L+1)), sum(CF_col_sections)), dtype=float) if sparse else np.zeros((D * 2**(D*(L+1)), sum(CF_col_sections)))
CF_fake_inverse = csr_matrix((sum(CF_col_sections), (D * 2**(D*(L+1)))), dtype=float) if sparse else np.zeros((sum(CF_col_sections), D * 2**(D*(L+1))))
for l in range(1,L+1):
    C_l = FEM.getC_l(D, l)
    
    T_squiggle = FEM.get_T_squiggle(D, l, L)
    level_weight = 2 ** (-l * (2-D) / 2)
    CFl = level_weight * T_squiggle @ C_l # section s corresponding to level l of the CF matrix

    CF[:,sum(CF_col_sections[:l-1]):sum(CF_col_sections[:l])] = CFl.toarray()
    CF_fake_inverse[sum(CF_col_sections[:l-1]):sum(CF_col_sections[:l]),:] = np.linalg.pinv(CFl.toarray())

CF_inv = np.round(np.linalg.pinv(CF.toarray() if sparse else CF), decimals=5)
CF_inversion_check = np.round(CF_fake_inverse @ CF, decimals=5)
# set up the diffusion coefficient matrix
#mat_L = 2
#diffusion_mat_small = np.diag(np.random.rand(2**(D*mat_L))) # matrix must be 2^(D*mat_L) X 2^(D*mat_L)
#diffusion_mat = np.kron(diffusion_mat_small, np.eye(2**(L - mat_L)))
#diffusion_mat = np.diag(np.random.rand(2**(D*L))) # matrix must be 2^(D*L) X 2^(D*L)
#diffusion_mat = np.diag(3*np.ones(2**(D*L)))
diffusion_mat = np.diag(np.concatenate((3*np.ones(2**(D*L-1)),5*np.ones(2**(D*L-1))))) # matrix of diffusion coefficients
D_A = np.kron(diffusion_mat, np.eye(D))


############## Assess errors between the two methods of creating the preconditioned system ##########################

CF_error = C_F_test - CF
print("CF error: ", sp.sparse.linalg.norm(CF_error) if sparse else np.linalg.norm(CF_error))

S = np.transpose(C_l) @ np.kron(D_A, np.eye(2**D)) @ C_l
#F_test = np.linalg.pinv(C_L.toarray()) @ CF # The preconditioner F matrix if we assume that we created CF correctly
#CF_test2 = C_l @ F_test
FSF1 = np.transpose(F) @ S @ F # preconditioned system using the F matrix
FSF2 = np.transpose(CF) @ np.kron(D_A, np.eye(2**D)) @ CF # preconditioned system using the CF matrix (should be the same as FSF1)

# Test if we can use the fast inversion of a normal matrix theorem to invert the FSF system
normal_test_1 = FSF2 @ np.transpose(FSF2)
normal_test_2 = np.transpose(FSF2) @ FSF2
normal_test_diff = normal_test_1 - normal_test_2
FSF2_inv = np.linalg.pinv(FSF2)
FSF2_inv_test = np.transpose(CF) @ np.linalg.inv(np.kron(D_A, np.eye(2**D))) @ CF

FSF1_inv = np.linalg.pinv(FSF1)
FSF1_norm = np.linalg.norm(FSF1)
FSF1_inv_norm = np.linalg.norm(FSF1_inv)
FSF1_cond = FSF1_norm * FSF1_inv_norm
#print("FSF1 norm: ", FSF1_norm)
#print("FSF1_inv norm: ", FSF1_inv_norm)
#print("FSF1 cond: ", FSF1_cond)

S_inv = np.linalg.pinv(S)
S_norm = np.linalg.norm(S)
S_inv_norm = np.linalg.norm(S_inv)
S_cond = S_norm * S_inv_norm
#print("\nS norm: ", S_norm)
#print("S_inv norm: ", S_inv_norm)
#print("S cond: ", S_cond)

FSF_error_mat = FSF1 - FSF2
FSF_error = np.linalg.norm(FSF_error_mat)
print("FSF error: ", FSF_error)

print("done")



