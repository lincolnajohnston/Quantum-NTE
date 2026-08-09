import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# make a laplacian (containing constant values, no 1/h^2 scaling) with zero Dirichlet boundary conditions
def get_laplacian(N):
    return_mat = (2 * np.diag(np.ones(N)) - np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))
    return return_mat

# return a diagonal matrix with the Diffusion coefficients in D_vec and the locations of material changes in m_vec
def get_diagonal_xs_mat(xs_vec, m_vec):
    diag_vec = np.zeros(sum(m_vec))
    cur_pos = 0
    for i in range(len(xs_vec)):
        diag_vec[cur_pos:cur_pos + m_vec[i]] = xs_vec[i] * np.ones(m_vec[i])
        cur_pos += m_vec[i]
    return np.diag(diag_vec)

# get the harmonic average of diffusion coeffcients at the interface between two materials
def get_av_D(D_lower, D_upper, delta):
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)

# at the interfaces between different materials, use the edge-averaged diffusion coeffcients to create a 2x2 correction matrix to add to the L matrix
def apply_L_mat_correction(L_mat, D_vec, m_vec, h):
    j = 0
    for i in range(len(D_vec)-1):
        D_edge = get_av_D(D_vec[i], D_vec[i+1], h)
        C = np.array([[D_edge - D_vec[i]/h, -D_edge + D_vec[i]/h], [-D_edge + D_vec[i+1]/h, D_edge - D_vec[i+1]/h]])
        j += m_vec[i] # location in the L_mat to make the correction
        L_mat[j-1:j+1, j-1:j+1] += (1/(h)) * C
    return L_mat


domain_size = 1
n = 7
N = int(math.pow(2,n))
h = domain_size / N
D_vec = [7,9]
m_vec = [int(N/2),int(N/2)]
sigma_a_vec = [3,4]
nu_sigma_f_vec = [5,6]

# create the diffusion matrix
laplacian_mat = get_laplacian(N)
D_mat = get_diagonal_xs_mat(D_vec, m_vec)
L_mat = (1/(h*h)) * D_mat @ laplacian_mat # L_mat without the correction for the interface diffusion coefficient
L_mat = apply_L_mat_correction(L_mat, D_vec, m_vec, h)

# create the diagonal cross section matrices
sigma_a_mat = get_diagonal_xs_mat(sigma_a_vec, m_vec)
nu_sigma_f_mat = get_diagonal_xs_mat(nu_sigma_f_vec, m_vec)

A_mat = np.linalg.inv(L_mat + sigma_a_mat) @ nu_sigma_f_mat

_,L_sing_vals,_ = np.linalg.svd(L_mat) # I think I'll get the same result whether I do singular value sof eigenvalues because the matrix is symmetric and normal or whatever
print("Max L singular value: ", max(L_sing_vals))
print("Min L singular value: ", min(L_sing_vals))
print("L condition number: ", max(L_sing_vals) / min(L_sing_vals))
print("")

_,A_sing_vals,_ = np.linalg.svd(A_mat)
print("Max A singular value: ", max(A_sing_vals))
print("Min A singular value: ", min(A_sing_vals))
print("A condition number: ", max(A_sing_vals) / min(A_sing_vals))