import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# For a 1-D, 1-G diffusion problem, I want to see how the error of the eigenvalue 
# and eigenvector solution will scale with h when the approximation of C is used that is also O(h)
# assume constant h thoughout problem and 2-material domain

################ functions to find the L, T, and F matrices for varying h values, also a function to get the approximated L matrix using the inprecise correction matrix, C ################

# google AI created this function
def create_tridiagonal(n, a, b, c):
    """
    Creates an n x n tridiagonal matrix.

    Args:
        n: The size of the matrix (number of rows and columns).
        a: List of subdiagonal elements (length n-1).
        b: List of main diagonal elements (length n).
        c: List of superdiagonal elements (length n-1).

    Returns:
        A NumPy array representing the tridiagonal matrix.
    """
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = b[i]
        if i > 0:
            matrix[i, i - 1] = a[i - 1]
        if i < n - 1:
            matrix[i, i + 1] = c[i]
    return matrix

#return the edge-averaged diffusion coefficient (at the edge of a cell)
# which is the harmonic mean of the diffusion coefficient divided by the cell width, dx
def get_edge_av_diff_coef(D1, D2, dx1, dx2):
    return 2 * (D1/dx1) * (D2/dx2) / (D1/dx1 + D2/dx2)

# get the diffusion term matrix (without the correction matrix approximation)
# m is the index of the first element of the second material, D0 and D1 are the diffusion coefficients of materials 0 and 1
def get_L(D0, D1, m, n, h):
    D_01 = get_edge_av_diff_coef(D0, D1, h, h)
    tridiag = 1/(h*h) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    D_mat = np.diag(np.concatenate((D0*np.ones(m),D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    L = D_mat @ tridiag
    C_mat = np.array([[-D0/(h*h) + D_01/h,D0/(h*h)-D_01/h],[D1/(h*h)-D_01/h,-D1/(h*h) + D_01/h]]) # each element in this 2x2 matrix scales as O(1/h^2)
    L[m-1:m+1,m-1:m+1] += C_mat
    return L

# get the inverse of the L matrix, if approx=true, assume the VA^(-1)U term is zero in the Woodbury matrix approximation
# k is the constant to add to the diagonals of the C matrix (taken from the absorption cross section matrix, T) to avoid a singular matrix
def get_L_inv(D0, D1, m, n, h, k, approx=False):
    D_01 = get_edge_av_diff_coef(D0, D1, h, h)
    tridiag = 1/(h*h) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    tridiag_inv = np.linalg.inv(tridiag) # O(log(n)) in quantum computer using disgonalization, DCT
    D_mat_inv = np.diag(np.concatenate((1/D0*np.ones(m),1/D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    A_inv = tridiag_inv @ D_mat_inv
    C_mat = np.array([[-D0/(h*h) + D_01/h + k,D0/(h*h)-D_01/h],[D1/(h*h)-D_01/h,-D1/(h*h) + D_01/h + k]]) # each element in this 2x2 matrix scales as O(1/h^2)
    C_mat_inv = np.linalg.inv(C_mat)

    if approx:
        P_exact_mat_inv = C_mat_inv + A_inv[m-1:m+1,m-1:m+1]
        P_exact_mat = np.linalg.inv(P_exact_mat_inv)

        P_mat = C_mat
        P_mat_error = P_mat - P_exact_mat
        P_mat_inv_error = C_mat_inv - P_exact_mat_inv
        print("P_mat_inv_error: ", P_mat_inv_error)
        print("P_mat_error: ", P_mat_error)
    else:
        P_mat_inv = C_mat_inv + A_inv[m-1:m+1,m-1:m+1]
        P_mat = np.linalg.inv(P_mat_inv)
    P_mat_pad = np.zeros((n,n))
    P_mat_pad[m-1:m+1,m-1:m+1] = P_mat

    L = A_inv - A_inv @ P_mat_pad @ A_inv
    return L


def get_T(sigma_a_0, sigma_a_1, m, n, k, h):
    sigma_a_mat = np.diag(np.concatenate((sigma_a_0*np.ones(m),sigma_a_1*np.ones(n-m))))
    sigma_a_mat[m-1,m-1] -= k
    sigma_a_mat[m,m] -= k
    return sigma_a_mat

def get_F(nu_sigma_f_0, nu_sigma_f_1, m, n, h):
    sigma_f_mat = np.diag(np.concatenate((nu_sigma_f_0*np.ones(m),nu_sigma_f_1*np.ones(n-m))))
    return sigma_f_mat


D0 = 2
sigma_a_0 = 7
nu_sigma_f_0 = 9

D1 = 4
sigma_a_1 = 5
nu_sigma_f_1 = 1

n_list = np.power(2, range(3,10), dtype=int)
m_list = n_list / 4
h_list = 1/n_list

################ solve the discretized system for different h values, show eigenvector and eigenvalue scaling with and without the approximation of C to O(h) error ################

k_list_original = np.zeros(len(n_list))
k_list_approx = np.zeros(len(n_list))
for n_i, n in enumerate(n_list):
    h = h_list[n_i]
    m = int(m_list[n_i])
    k = 1 # value to be removed from the absorption cross section diagonal matrix and inserted into the diffusion matrix to avoid a singular correction submatrix, C

    L_original = get_L(D0, D1, m, n, h)
    L_inv_original = get_L_inv(D0, D1, m, n, h, k, approx=False)
    T_original = get_T(sigma_a_0, sigma_a_1, m, n, k, h) # absorption cross section matrix after having k removed from the material boundary cells
    F_original = get_F(nu_sigma_f_0, nu_sigma_f_1, m, n, h) # nu-fission cross section matrix

    L_inv_approx = get_L_inv(D0, D1, m, n, h, k, approx=True)
    T_approx = get_T(sigma_a_0, sigma_a_1, m, n, k, h)
    F_approx = get_F(nu_sigma_f_0, nu_sigma_f_1, m, n, h)

    L_inv_error = L_inv_approx - L_inv_original
    T_error = T_original - T_approx

    '''A_mat_1 = np.linalg.inv(L_original + T_original) @ F_original
    eigenvalues1, eigenvectors1 = np.linalg.eig(A_mat_1)
    print("eig 1: ")
    print(eigenvalues1)
    print(eigenvectors1)'''

    A_mat_original = np.linalg.inv(np.eye(n) + L_inv_original @ T_original) @ L_inv_original @ F_original
    eigenvalues_original, eigenvectors_original = np.linalg.eig(A_mat_original)
    max_eig_original = max(eigenvalues_original)
    k_list_original[n_i] = max_eig_original
    print("original eigen values/vectors: ")
    #print(eigenvalues_original)
    #print(eigenvectors_original)

    A_mat_approx = np.linalg.inv(np.eye(n) + L_inv_approx @ T_approx) @ L_inv_approx @ F_approx
    eigenvalues_approx, eigenvectors_approx = np.linalg.eig(A_mat_approx)
    max_eig_approx = max(eigenvalues_approx)
    k_list_approx[n_i] = max_eig_approx
    print("approximate eigen values/vectors: ")
    #print(eigenvalues_approx)
    #print(eigenvectors_approx)

best_k_eig_guess = k_list_original[-1]
k_error_list_original = k_list_original - best_k_eig_guess
k_error_list_approx = k_list_approx - best_k_eig_guess


plt.loglog(h_list, k_error_list_original)
plt.loglog(h_list, k_error_list_approx)
plt.title("k-eig error vs h")
plt.legend(["original", "approx"])
plt.show()