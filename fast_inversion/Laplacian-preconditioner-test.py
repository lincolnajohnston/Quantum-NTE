import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.fftpack import dct, dst
import math

# Goal of this script is to determine the scaling of the condition number and matrix norm of the matrix
# M = L^(-1/2) A L^(-1/2). Where L is the unmodified Laplacian matrix with 0 Dirichlet boundary conditions
# and A is the diffusion operator in the discretized diffusion equation with piecewise constant
# diffucion coefficients
# If the block-encoding factor grows fast with the grid size N, then M cannot be applied to a quantum computer efficiently
# If the condition number of M grows fast with N, then inverting M on a quantum computer will be inefficient, which needs
# to be done to effectively apply M^(-1) through this preconditioning method

# conclusion: The matrix norms and the condition numbers of the M matrix seem to be O(1) w.r.t. N, but the matrix
# norms and condition numbers of the P matrix (L^(-1)A) seem to be O(N), maybe O(logN), but probably not
# So we do have to use the M matrix, not the P matrix, but it is unclear how we can block encode the M matrix
# efficiently because M will be a dense matrix, unlike P

# Just considering the 1D Laplacian for now

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

n_list = np.array([8,16,32,64,80, 100, 110, 128, 150, 180, 220, 256, 300, 350, 400, 450, 512, 600, 700, 800, 900, 1024, 1100, 1200, 1300, 1400, 1500, 1600])
h = 1/n_list
k = 2 # size of small matrix (correction for boundary between materials)
m = 4 # location of material boundary (first cell of new material)
D0 = 7
D1 = 8

M_conds_list = np.zeros(len(n_list))
P_conds_list = np.zeros(len(n_list))
M_norms_list = np.zeros(len(n_list))
P_norms_list = np.zeros(len(n_list))
for n_i,n in enumerate(n_list):
    D_01 = get_edge_av_diff_coef(D0, D1, h[n_i], h[n_i])
    h2 = h[n_i]*h[n_i]
    L = 1/(h2) * create_tridiagonal(n,-1*np.ones(n-1), 2*np.ones(n), -1*np.ones(n-1)) # O(log(n))
    F = create_tridiagonal(n+1,0*np.ones(n),1*np.ones(n+1),-1*np.ones(n))
    F = F[:-1,:]

    # create the F matrix with the DCT and DST matrices
    DCT_2 = dct(np.eye(n+1), type=2, norm="ortho")
    DST_1 = dst(np.eye(n), type=1, norm="ortho")
    F_lambdas = np.diag((2 / h[n_i]) * np.sin(math.pi / (2 * (n+1)) * np.array(range(1,n+1))))
    F = DST_1 @ F_lambdas @ np.transpose(DCT_2[:,1:])

    # create the F matrix with the DCT and DST matrices
    DCT_2_n1 = dct(np.eye(n+1), type=2, norm="ortho")
    DST_n = dst(np.eye(n), type=1, norm="ortho")
    DST_1_test = np.zeros((n+1, n+1))
    DST_1_test[:n,:n] = DST_n
    lambda_list = list((2 / h[n_i]) * np.sin(math.pi / (2 * (n+1)) * np.array(range(0,n)))) + [0]
    F_lambdas_test = np.diag(lambda_list)
    F = DST_1_test @ F_lambdas_test @ np.transpose(DCT_2_n1)

    # Test creating the Neumann BC Laplacian from DCT-2
    #test_lambdas = np.diag((1 / h2) * (2 - 2 * np.cos(math.pi * np.array(range(n)) / n)))
    #test_laplacian_neumann = DCT_2 @ test_lambdas @ np.linalg.inv(DCT_2)


    D_F_mat = np.diag(np.concatenate((D0*np.ones(m),np.array([D_01*h[n_i]]), D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in terms of n
    A_mat_test = (1/h2) * F @ D_F_mat @ np.transpose(F)

    D_mat = np.diag(np.concatenate((D0*np.ones(m),D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in terms of n
    D_mat_inv = np.diag(np.concatenate((1/D0*np.ones(m),1/D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in terms of n
    A_mat = D_mat @ L # O(Mlog(n)) to encode quantumly

    C_mat = np.array([[-D0/h2 + D_01/h[n_i],D0/h2-D_01/h[n_i]],[D1/h2-D_01/h[n_i],-D1/h2 + D_01/h[n_i]]]) # each element in this 2x2 matrix scales as O(1/h^2)

    A_mat[m-1:m+1,m-1:m+1] += C_mat
    A_diff = A_mat_test - A_mat
    L_inv_mat = np.linalg.inv(L)
    L_sqrt_inv_mat = sqrtm(L_inv_mat)
    M_mat = L_sqrt_inv_mat @ A_mat @ L_sqrt_inv_mat
    P_mat = L_inv_mat @ A_mat

    # print condition number and matrix norm of L, A, and L^(-1/2) A L^(-1/2)
    L_cond = np.linalg.cond(L)
    A_cond = np.linalg.cond(A_mat)
    M_cond = np.linalg.cond(M_mat)
    P_cond = np.linalg.cond(P_mat)
    M_conds_list[n_i] = M_cond
    P_conds_list[n_i] = P_cond

    L_norm = np.linalg.norm(L, ord=2)
    A_norm = np.linalg.norm(A_mat, ord=2)
    M_norm = np.linalg.norm(M_mat, ord=2)
    P_norm = np.linalg.norm(P_mat, ord=2)
    M_norms_list[n_i] = M_norm
    P_norms_list[n_i] = P_norm

    print("----------n = ", n, "----------")
    print("L condition number: ", L_cond)
    print("A condition number: ", A_cond)
    print("M condition number: ", M_cond)
    print("P condition number: ", P_cond, "\n")

    print("L matrix norm: ", L_norm)
    print("A matrix norm: ", A_norm)
    print("M matrix norm: ", M_norm)
    print("P matrix norm: ", P_norm)
    print("-------------------------------\n")

plt.plot(n_list, M_conds_list)
plt.plot(n_list, P_conds_list)
plt.legend(["M condition numbers","P condition numbers"])
plt.figure()

plt.plot(n_list, M_norms_list)
plt.plot(n_list, P_norms_list)
plt.legend(["M matrix norms","P matrix norms"])
plt.show()