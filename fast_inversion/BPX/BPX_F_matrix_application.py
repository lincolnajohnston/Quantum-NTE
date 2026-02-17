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

# Decompose the F matrix into each indiviual matrix that needs to be applied on a quantum circuit, and
# show that each of these can be applied efficiently (referencing the method used to apply a matrix if it is not shown directly)

# return the y value (as fraction of max height) at position x for a triangle wave starting at x_min and ending at x_max
def triangle_wave(x: float, x_min, x_max) -> float:
    if x < x_min or x > x_max:
        return 0
    dx = x_max - x_min
    r = x - x_min
    return 1.0 - 2.0 * abs(r - dx/2) / dx

# get the basis change matrix from mutlilevel basis to the basis of the finest level
def get_F(D, L):
    n_fine = int(math.pow(2,L)) # number of points in finest level
    h_fine = 1/n_fine
    CF_col_sections = [(2**l-1)**D for l in range(1,L+1)] # list of number of basis functions in each level

    # create the 1-D F matrix preconditioner (defined at the bottom of page 11 of the Deiml paper)
    F = csr_matrix(((n_fine-1)**D, sum(CF_col_sections)), dtype=float)
    fine_position_vals = [np.linspace(h_fine,1-h_fine,n_fine-1) for i in range(D)]
    position_meshes = np.meshgrid(*fine_position_vals, indexing='ij')
    position_points = [position_meshes[i].flatten() for i in range(D)]
    col = 0
    for l in range(1,L+1):
        n_coarse = int(math.pow(2,l)) # number of basis functions in current level
        h_coarse = 1/n_coarse
        triangle_height_weight = 1 # height of hat function in basis function
        level_weight = 2 ** (-l * (2-D) / 2)
        #level_weight = 1
        for i in range((n_coarse - 1)**D): # find values of current column of F
            pos_indices = [int(i%(n_coarse-1)**(D-d) / (n_coarse-1)**(D-d-1)) for d in range(D)]
            pos = [position_points[d][i] for d in range(D)]
            weights_list = np.array([[triangle_height_weight * triangle_wave(x,h_coarse*pos_indices[d],h_coarse*(pos_indices[d]+2)) for x in fine_position_vals[d]] for d in range(D)])
            #weights_list = [[triangle_height_weight * triangle_wave(x,h_coarse*pos_indices[d],h_coarse*(pos_indices[d]+2)) for x in fine_position_vals[d]] for d in range(D)]
            weights_nonzero_indices = [np.nonzero(weights_list[d])[0] for d in range(D)]
            for combo in itertools.product(*weights_nonzero_indices):
                row = int(np.sum([combo[d] * np.prod([len(weights_list[d_p]) for d_p in range(d+1,D)]) for d in range(D)]))
                weight = [weights_list[d,combo[d]] for d in range(D)]
                F[row,col] =  level_weight * np.prod(weight)
            col += 1
    return F

# return the F_prime matrix which does the multilevel to single level basis change without scaling each level by a constant
def get_F_prime(D, L):
    n_fine = int(math.pow(2,L)) # number of points in finest level
    h_fine = 1/n_fine
    CF_col_sections = [(2**l-1)**D for l in range(1,L+1)] # list of number of basis functions in each level

    # create the 1-D F matrix preconditioner (defined at the bottom of page 11 of the Deiml paper)
    F = csr_matrix(((n_fine-1)**D, sum(CF_col_sections)), dtype=float)
    fine_position_vals = [np.linspace(h_fine,1-h_fine,n_fine-1) for i in range(D)]
    position_meshes = np.meshgrid(*fine_position_vals, indexing='ij')
    position_points = [position_meshes[i].flatten() for i in range(D)]
    col = 0
    for l in range(1,L+1):
        n_coarse = int(math.pow(2,l)) # number of basis functions in current level
        h_coarse = 1/n_coarse
        triangle_height_weight = 1 # height of hat function in basis function
        level_weight = 1
        for i in range((n_coarse - 1)**D): # find values of current column of F
            pos_indices = [int(i%(n_coarse-1)**(D-d) / (n_coarse-1)**(D-d-1)) for d in range(D)]
            pos = [position_points[d][i] for d in range(D)]
            weights_list = np.array([[triangle_height_weight * triangle_wave(x,h_coarse*pos_indices[d],h_coarse*(pos_indices[d]+2)) for x in fine_position_vals[d]] for d in range(D)])
            #weights_list = [[triangle_height_weight * triangle_wave(x,h_coarse*pos_indices[d],h_coarse*(pos_indices[d]+2)) for x in fine_position_vals[d]] for d in range(D)]
            weights_nonzero_indices = [np.nonzero(weights_list[d])[0] for d in range(D)]
            for combo in itertools.product(*weights_nonzero_indices):
                row = int(np.sum([combo[d] * np.prod([len(weights_list[d_p]) for d_p in range(d+1,D)]) for d in range(D)]))
                weight = [weights_list[d,combo[d]] for d in range(D)]
                F[row,col] =  level_weight * np.prod(weight)
            col += 1
    return F

# return the scaling matrix, C, such that F = F_prime @ C
def get_C(D, L):
    n_fine = int(math.pow(2,L)) # number of points in finest level
    h_fine = 1/n_fine
    CF_col_sections = [(2**l-1)**D for l in range(1,L+1)] # list of number of basis functions in each level

    C = csr_matrix((sum(CF_col_sections), sum(CF_col_sections)), dtype=float)
    col = 0
    for l in range(1,L+1):
        n_coarse = int(math.pow(2,l)) # number of basis functions in current level
        h_coarse = 1/n_coarse
        level_weight = 2 ** (-l * (2-D) / 2)
        for i in range((n_coarse - 1)**D): # find values of current column of F
            C[col, col] = level_weight
            col += 1
    return C

# return the Fu matrix, which only has the ones of the F matrix
def get_F_u(L):
    section_length_list = [int(math.pow(2,n_p)-1) for n_p in range(1,L+1)]
    N = int(math.pow(2,L))
    Fu = np.zeros((N-1,sum(section_length_list)))
    for s in range(L):
        col_offset = sum(section_length_list[:s])
        row_jump = 2**(L-s-1)
        row_offset = row_jump - 1
        section_size = section_length_list[s]
        for col in range(section_size):
            Fu[row_offset + col*row_jump, col_offset + col] = 1
    return Fu

# returns the F_{u,l} matrix, which only has the ones of section l of the F matrix
def get_F_us_1D(L, s):
    section_length_list = [int(math.pow(2,n_p)-1) for n_p in range(1,L+1)]
    N = int(math.pow(2,L))
    Fu = np.zeros((N,N))
    row_jump = 2**(L-s-1)
    row_offset = row_jump - 1
    section_size = section_length_list[s]
    for col in range(section_size):
        Fu[row_offset + col*row_jump, col] = 1
    return Fu

# returns the F_{u,l} matrix, which only has the ones of section l of the F matrix
def get_F_us(L, s):
    section_length_list = [int(math.pow(2,n_p)-1) for n_p in range(1,L+1)]
    N = int(math.pow(2,L))
    Fu = np.zeros((N-1,sum(section_length_list)))
    col_offset = sum(section_length_list[:s])
    row_jump = 2**(L-s-1)
    row_offset = row_jump - 1
    section_size = section_length_list[s]
    for col in range(section_size):
        Fu[row_offset + col*row_jump, col_offset + col] = 1
    return Fu

# return the column permutation matrix used for multidimensional F matrices
# TODO: fill in the rest of the permutation matrix so that it is unitary
def get_P_c(L,D,s):
    P_c = np.zeros((int(2**(L*D)),int(2**(L*D))))
    N = int(2**(L))
    N_s = int(2**(s+1) - 1)
    #offsets = [sum([N**d_p * math.floor(i%(N_s**(d_p+1)) / N_s**d_p) for d_p in range(D)]) for i in range(N_s**D)] # only fill in the relevant columns (first N_s*D)
    offsets = [sum([N**d_p * math.floor(i%(N_s**(d_p+1)) / N_s**d_p) for d_p in range(D)]) for i in range(int(2**(L*D)))] # fill in all columns (what gate might actually be block-encoded in a circuit)
    for col, row in enumerate(offsets):
        P_c[row, col] = 1
    return P_c

# return the row permutation matrix used for multidimensional F matrices
# TODO: fill in the rest of the permutation matrix so that it is unitary
def get_P_r(L,D,s):
    P_r = np.zeros((int(2**(L*D)),int(2**(L*D))))
    N = int(2**(L))
    #offsets = [sum([N**(d_p) * math.floor(i%((N-1)**(d_p+1)) / (N-1)**d_p) for d_p in range(D)]) for i in range((N-1)**D)] # only fill in the relevant columns (first N_s*D)
    offsets = [sum([N**(d_p) * math.floor(i%((N-1)**(d_p+1)) / (N-1)**d_p) for d_p in range(D)]) for i in range(int(2**(L*D)))] # fill in all columns (what gate might actually be block-encoded in a circuit)
    for row, col in enumerate(offsets):
        P_r[row, col] = 1
    return P_r

# return the column permutation matrix that shifts F_us from the leftmost column to its proper place within the F_u matrix
def get_P_s(L,D,s):
    section_length_list = [int((2**n_p-1)**D) for n_p in range(1,L+1)]
    offset = sum(section_length_list[:s])
    P_s = np.diag(np.ones(2**(D*(L+1)) - offset), k=offset)
    return P_s

# dilator matrix that does the transformation E|x> = 0.5|(x-offset) mod N> + 1|x> + 0.5|(x+offset) mod N>
def get_E(N, offset = 1):
    matrix = np.zeros((N, N))
    matrix[0:N,0:N] += np.diag(np.ones(N)) # diagonal terms, identity matrix O(1) time to apply

    # |x> -> 0.5|(x-offset) mod N>
    # |x> -> 1|(x-offset) mod N> is unitary and can be implemented in O(polylog(N) time)
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=offset)  # |x> -> |x-offset> term when x >= offset
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=offset-N) # |x> -> |x-offset> term when x < offset

    # |x> -> 0.5|(x+offset) mod N>
    # |x> -> 1|(x+offset) mod N> is unitary and can be implemented in O(polylog(N) time)
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=N-offset)  # |x> -> |x+offset> term when x >= N - offset
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=-offset) # |x> -> |x+offset> term when x < N - offset

    # Each of the three terms can be implemented as separate unitary matrices. Then LCU can be used to sum them, with alpha values of 0.5, 1, and 0.5
    # the time complexity of doing LCU is O(alpha) (which I think comes from the post-selection procedure, which you can do amplitude amplification for 
    # which takes O(alpha) rotations to get a high probability of success).
    # So alpha=2, E/2 can be block-encoded in O(polylog(N) + alpha) = O(polylog(N)) time

    return matrix

# the domain goes from 0 to 1
D_min = 2
D_max = 2
L_min = 3
L_max = 8
D_vals = np.array(range(D_min,D_max + 1))
L_vals = np.array(range(L_min,L_max + 1)) # number of levels of BPX preconditioner
D_and_L = [D_vals, L_vals]

mat_L = 3
diffusion_mat_small = [np.diag(np.random.rand(2**(D*mat_L))) for D in D_vals]

for combo in itertools.product(*D_and_L):
    D = combo[0]
    L = combo[1]
    # find C_L (C_l for the finest level)
    section_length_list = [int((2**n_p-1)**D) for n_p in range(1,L+1)]
    F = get_F(D, L)

    F_u = np.zeros((2**(D*(L+1)), 2**(D*(L+1))))
    for s in range(L):
        F_us_1D = get_F_us_1D(L, s) # can be constructed with transformation rules like a permutation matrix
        F_us = np.ones((1,1))
        for d in range(D):
            F_us = np.kron(F_us, F_us_1D) # kroncker product of the F_us_1D matrices

        # permutation matrices with constant number of equations to describe the transformation O(polylogN)
        P_c = get_P_c(L, D, s) # first column permutation
        P_r = get_P_r(L, D, s) # row permutation
        P_s = get_P_s(L, D, s) # second column permutation to shift the F_us matrix to its proper location within F_u

        F_us = np.kron(np.eye(int(2**D)), P_r @ F_us @ P_c) @ P_s # apply permutation matrices to F_us

        # Create the dilator matrix, E
        E = np.eye(2**(D*(L)))
        for s_p in range(L-s-1, 0 ,-1):
            E_p_1D = get_E(2**((L)), offset=2**(s_p-1)) # 1D dilator matrices are tridiagonal Toeplitz matrices, can be applied in O(polylogN) time
            E_p = np.array([1]) # higher dimensional dilator matrices are just kronecker products of the lower dimensional dilator matrices, O(d polylog(N))
            for d in range(D):
                E_p = np.kron(E_p, E_p_1D)
            E = E @ E_p
        E = P_r @ E @ np.transpose(P_r) # apply permutations on the dilator matrix for higher-dimensional cases
        E = np.kron(np.eye(2**D), E) # increase size of dilator matrix to match size of F_us
        F_us = E @ F_us # Dilate the F_us matrix

        F_u += F_us # combine F_us together with LCU
        F_us_norm = np.linalg.norm(F_us, ord=2)
        print("F_u"+str(s)+" norm: " + str(F_us_norm))

    F_prime = get_F_prime(D,L)
    F_prime_shape = F_prime.get_shape()
    F_diff = F_prime.toarray() - F_u[:F_prime_shape[0],:F_prime_shape[1]]

    C = get_C(D, L)
    F_2 = F_prime @ C
    F_prime_norm = np.linalg.norm(F_prime.toarray(), ord=2)
    C_norm = np.linalg.norm(C.toarray(), ord=2)
    F_2_norm = np.linalg.norm(F_2.toarray(), ord=2)

    F_prime_cond = np.linalg.cond(F_prime.toarray())
    C_cond = np.linalg.cond(C.toarray())
    F_2_cond = np.linalg.cond(F_2.toarray())

    print("F norm: ", F_prime_norm)
    print("C norm: ", C_norm)
    print("F_2 norm: ", F_2_norm)

    print("\n F cond: ", F_prime_cond)
    print("C cond: ", C_cond)
    print("F_2 cond: ", F_2_cond)
