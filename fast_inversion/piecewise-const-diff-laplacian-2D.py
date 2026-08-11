import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# Create the matrices needed to apply the inverse of the diffusion term in the diffusion equation
# Assume a problem with 4 materials in rectangular regions defined along a Cartesian grid, constant D and sigma_a within each 
# material, and constant h (cell width) throughout the problem domain

# ChatGPT function to get a 1-D array along a certain axis
def iter_slices(arr: np.ndarray, axis: int):
    """
    Yield every 1-D slice taken along `axis` in `arr`, together with the
    index tuple of the other axes.

    Example: arr.shape == (4, 5, 6), axis == 1
             yields ( (i, k), arr[i, :, k] ) for all i, k.
    """
    # bring the axis you care about to the last position
    moved = np.moveaxis(arr, axis, -1)          # shape (..., L)

    # Flatten every axis *except* the last one into a single dimension
    lead, length = moved.shape[:-1], moved.shape[-1]
    flat = moved.reshape(-1, length)            # shape (prod(lead), L)

    # Iterate.  Use np.unravel_index to get the multi-dimensional index
    for flat_idx, vec in enumerate(flat):
        other_axes_idx = np.unravel_index(flat_idx, lead)
        yield other_axes_idx, vec


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


#n_list = np.array([1]) # big matrix size
#n_list = np.array(range(10,300,10))
D = 2 # 2D
n = np.array([8,8])
h = 1/n
n_mats = np.array([4,4]) # number of material regions in each dimension, assume each material region is exactly the same size
#m = [4,2] # location of material boundary (first cell of new material)
D_list = np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18],[19,20,21,22]])
sigma_a_list = np.array([[27,28,29,30],[31,32,33,34],[35,36,37,38],[39,40,41,42]])
k_list = np.array([[0.5, 0.5, 0.5, 0.5],[0.5, 0.5, 0.5, 0.5],[0.5, 0.5, 0.5, 0.5],[0.5, 0.5, 0.5, 0.5]])

'''D = 3 # 3D
n = np.array([8,8,8])
h = 1/n
n_mats = [2,2,2] # number of material regions in each dimension, assume each material region is exactly the same size
#m = [4,2] # location of material boundary (first cell of new material)
D_list = [[[7,8],[9,10]],[[11,12],[13,14]]]
sigma_a_list = [[[27,28],[29,30]],[[31,32],[33,34]]]
k_list = [[[0.5, 0.5], [0.5, 0.5]],[[0.5, 0.5], [0.5, 0.5]]]'''

M_total = np.zeros(np.power(n,D))
D_total = np.zeros((np.prod(n), np.prod(n)))

# create the diffusion coefficient matrix
for idx, vec in iter_slices(D_list, axis=D-1):
    idx = np.flip(idx) # I use idx in (x,y,z) format but the data and cross sections are in (z,y,x) format (meaning z is most significant)
    print("idx: ", idx)
    print("vec: ", vec)

    D_bar = np.diag(np.kron(vec,np.ones(int(n[0]/n_mats[0]))))
    # Diffusion coefficient matrix, D
    for d_i in range(1,D): # iterate through non-current dimensions
        sub_mat_term = np.zeros((n_mats[d_i],n_mats[d_i])) # sub_mat_term is the outer product between a ket and a bra |x><x|
        sub_mat_term[idx[d_i-1],idx[d_i-1]] = 1
        D_bar = np.kron(sub_mat_term, np.kron(np.eye(int(n[d_i]/n_mats[d_i])), D_bar))
    D_total += D_bar


# iterate over spatial dimensions, 0 is x, 1 is y etc
for d in range(D):
    # iterate over material rows in every spatial direction except for the current one, starts from most significant dimension
    for idx, vec in iter_slices(D_list, axis=D-d-1):
        # correction matrix stuff
        C_pad = np.zeros((n[d],n[d]))
        for m_i in range(n_mats[d]-1):
            shortened_index = np.array(idx)
            full_index_lower = np.insert(shortened_index,d+1,m_i) # indices are in z,y,x order
            full_index_upper = np.insert(shortened_index,d+1,m_i+1)

            D_lower = D_list[tuple(full_index_lower)]
            D_upper = D_list[tuple(full_index_upper)]
            sigma_a_lower = sigma_a_list[tuple(full_index_lower)]
            sigma_a_upper = sigma_a_list[tuple(full_index_upper)]
            k_lower = k_list[tuple(full_index_lower)]
            k_upper = k_list[tuple(full_index_upper)]

            D_avg = get_edge_av_diff_coef(D_lower, D_upper, h[d], h[d])
            C_mat = np.array([[-D_lower/(h[d]*h[d]) + D_avg/h[d] + k_lower*sigma_a_lower,D_lower/(h[d]*h[d])-D_avg/h[d]],[D_upper/(h[d]*h[d])-D_avg/h[d],-D_upper/(h[d]*h[d]) + D_avg/h[d] + k_upper*sigma_a_upper]]) # each element in this 2x2 matrix scales as O(1/h^2)
            m = (m_i+1) * int(n[d]/n_mats[d]) # index of start of new material
            C_pad[m-1:m+1,m-1:m+1] = C_mat
        print(C_pad)

        # TODO: add the C_pad together for each spatial dimension to create C_x, C_y, etc, then add these corections to the pure Laplacian term
        # This section is copypasted, no real work done here yet, can delete
        for d_i in range(0,D): # iterate through non-current dimensions
            if d_i == d:
                continue
            sub_mat_term = np.zeros((n_mats[d_i],n_mats[d_i])) # sub_mat_term is the outer product between a ket and a bra |x><x|
            sub_mat_term[idx[d_i-1],idx[d_i-1]] = 1
            D_bar = np.kron(sub_mat_term, np.kron(np.eye(int(n[d_i]/n_mats[d_i])), D_bar))
        D_total += D_bar

    '''other_dim_indices = [1]
    for d_i in range(D):
        if d == d_i:
            continue
        other_dim_indices = other_dim_indices + np.array(range(n_mats[d_i]))
    for m_i in other_dim_indices:
        D_vec = D_list[m_i]'''
    #D_01 = get_edge_av_diff_coef(D0, D1, h, h)
    L = 1/(h[d]*h[d]) * create_tridiagonal(n[d], -1*np.ones(n[d]-1), 2*np.ones(n[d]), -1*np.ones(n[d]-1)) # O(log(n))
    L_inv = np.linalg.inv(L) # O(log(n)), can do DCT/DST matrices inverses in same time as forward operators

    '''D_mat = np.diag(np.concatenate((D0*np.ones(m),D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    D_mat_inv = np.diag(np.concatenate((1/D0*np.ones(m),1/D1*np.ones(n-m)))) # O(M), M is number of materials, O(1) in term of n
    A_mat = D_mat @ L # O(Mlog(n)) to encode quantumly
    A_mat_inv = D_mat_inv @ L_inv # O(Mlog(n)) to encode quantumly

    sigma_a_mat = np.diag(np.concatenate((sigma_a_0*np.ones(m),sigma_a_1*np.ones(n-m))))
    sigma_a_mat_prime = sigma_a_mat.copy()
    sigma_a_mat_prime[m-1,m-1] *= (1-k_0)
    sigma_a_mat_prime[m,m] *= (1-k_1)

    V_mat = np.zeros((2,n))
    U_mat = np.zeros((n,2))
    for i in range(k):
        V_mat[i,m+i] = 1 
        U_mat[m+i,i] = 1
    A_inv_sub = V_mat @ A_mat_inv @ U_mat

    C_mat = np.array([[-D0/h2 + D_01/h[n_i] + k_0*sigma_a_mat[m-1,m-1],D0/h2-D_01/h[n_i]],[D1/h2-D_01/h[n_i],-D1/h2 + D_01/h[n_i] + k_1*sigma_a_mat[m,m]]]) # each element in this 2x2 matrix scales as O(1/h^2)
    C_mat_inv = np.linalg.inv(C_mat) # elements in this matrix are O(1) w.r.t. h as h->0

    P_mat_inv = C_mat_inv + A_inv_sub
    P_mat = np.linalg.inv(P_mat_inv)

    P_C_dif_mat = P_mat - C_mat
    print(P_C_dif_mat)

    Q = (U_mat@P_mat@V_mat) @ A_mat_inv
    R = A_mat_inv @ Q
    R_norm_L1 = np.max(R)
    R_norm_L2 = np.linalg.norm(R)
    R_norm_max_sing = np.linalg.norm(R, ord=2)

    L_final_mat = A_mat_inv - R'''

print("done")
