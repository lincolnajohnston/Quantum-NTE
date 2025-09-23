import sys
import os
sys.path.append(os.getcwd())
from helpers.ProblemData import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# trying to visualize the matrices from "Quantum Realization of the Finite Element Method" by Deiml M, Peterseim D and make 
# sure they can be applied effectively and work as preconditioners

# return the y value (as fraction of max height) at position x for a triangle wave starting at x_min and ending at x_max
def triangle_wave(x: float, x_min, x_max) -> float:
    if x < x_min or x > x_max:
        return 0
    dx = x_max - x_min
    r = x - x_min
    return 1.0 - 2.0 * abs(r - dx/2) / dx

# return the C_{l,1D} matrix at the bottom of page 15 of the Deiml paper 
def get_C_l_1D(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[1,-1],[0,0]])) # TODO: figure out how this works, don't really understand this equation
    #I_l = np.eye(N)[:,:N-1]
    #N_l = np.eye(N)[:,1:N]
    #M2 = np.concatenate((I_l, N_l), axis=0)
    # M2 is now an operation performing |i> -> 1/sqrt(2) (|2i+1> + |2i+2>)
    M2 = np.zeros((2**(l+1), 2**l - 1))
    # Lincoln's M2
    for col in range(2**l - 1):
        M2[2*col+1, col] = 1
        M2[2*(col+1), col] = 1
    # Mahathi's M2
    '''for col in range(2**l - 1):
        M2[2*col, col] = 1
        M2[2*(col)+3, col] = 1'''
    return 2**(l/2) * M1 @ M2

# return the R_{l,1D} matrix at the bottom of page 15 of the Deiml paper 
def get_R_l_1D(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[1/2,1/2],[1/(2*math.sqrt(3)),-1/(2*math.sqrt(3))]]))
    I_l = np.eye(N)[:,:N-1]
    N_l = np.eye(N)[:,1:N]
    M2 = np.concatenate((I_l, N_l), axis=0)
    return 2**(-l/2) * M1 @ M2

# ChatGPT function, not fully tested
# return the Pi_l operator in the middle of page 15 of the Deiml paper
def jk_interleave_permutation_matrix(l: int, d: int, sparse: bool = True, dtype=np.uint8):
    """
    build the permutation matrix P such that for a state vector x ordered as
        |j1> ... |jd> |k1> ... |kd>
    (with |j_i| = l bits each, and each |k_i| = 1 bit),
    the product y = P @ x reorders amplitudes to the interleaved basis
        |j1>|k1> ... |jd>|kd>.

    Conventions:
    - Blocks keep their internal bit order; only whole blocks are interleaved.
    - Overall basis indexing is standard binary with bit 0 = least significant bit (rightmost).
    - Size is 2^n × 2^n, where n = d*(l+1).

    Parameters
    ----------
    l : int
        Number of bits in each j_i (l > 0).
    d : int
        Number of (j_i, k_i) pairs (d > 0).
    sparse : bool
        If True, return a scipy.sparse.csr_matrix. If False, return a dense np.ndarray.
    dtype : dtype
        Data type of the 1s in the permutation matrix (default uint8).

    Returns
    -------
    P : scipy.sparse.csr_matrix or np.ndarray
        Permutation matrix implementing the interleaving.
    """
    if l <= 0 or d <= 0:
        raise ValueError("l and d must be positive integers.")

    n = d * (l + 1)        # total bits
    N = 1 << n             # dimension

    # Input left-to-right blocks: [ J1(l) ... Jd(l) | K1 ... Kd ] (each K is a single bit)
    # Output left-to-right:       [ J1(l), K1, J2(l), K2, ..., Jd(l), Kd ]
    mapping_left = []
    J_starts = [i * l for i in range(d)]
    for i in range(d):
        # J_i block of length l
        mapping_left.extend(range(J_starts[i], J_starts[i] + l))
        # K_i is a single bit at left-position d*l + i in the input
        mapping_left.append(d * l + i)

    mapping_left = np.asarray(mapping_left, dtype=np.int64)  # length n

    # Convert left-to-right positions to bit indices (0 = LSB)
    out_left = np.arange(n, dtype=np.int64)
    dest_bit = n - 1 - out_left          # output bit indices (LSB-first)
    src_bit  = n - 1 - mapping_left      # corresponding input bit indices

    bit_src_for_dest = np.empty(n, dtype=np.int64)
    bit_src_for_dest[dest_bit] = src_bit

    # Build permutation of basis indices: for each input column x -> output row y
    rows = np.empty(N, dtype=np.int64)
    cols = np.arange(N, dtype=np.int64)
    for x in range(N):
        y = 0
        # Set each destination bit from the appropriate source bit of x
        for b in range(n):
            y |= ((x >> bit_src_for_dest[b]) & 1) << b
        rows[x] = y

    if sparse:
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("scipy is required for sparse output; install scipy or set sparse=False.") from e
        data = np.ones(N, dtype=dtype)
        return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=dtype)
    else:
        P = np.zeros((N, N), dtype=dtype)
        P[rows, cols] = 1
        return P

# ChatGPT function, not fully verified for correctness
# return the pi operator at the top of page 14 of the Deiml paper (except only acts on the first two registers)
def js_swap_perm_matrix(l: int, d: int, sparse: bool = True, dtype=np.uint8):
    """
    Return the permutation matrix P that maps |j>|s> -> |s>|j>,
    with |j| having 2^(d*l) basis states and |s| having d basis states.

    Dimensions:
      m = 2^(d*l)  (size of the j-register)
      n = d        (size of the s-register)
      N = m * n

    Basis ordering convention (standard Kronecker order):
      input index = j * n + s  (s varies fastest)
      output index = s * m + j

    Parameters
    ----------
    l : int
        Number of bits per j_i block; j has 2^(d*l) states total.
    d : int
        Number of (j_i, k_i) pairs earlier; here it's the size of |s|.
    sparse : bool
        If True, return a scipy.sparse.csr_matrix; else a dense ndarray.
    dtype : numpy dtype
        Storage type for 1s in the permutation.

    Returns
    -------
    P : scipy.sparse.csr_matrix or np.ndarray of shape (N, N)
    """
    if l <= 0 or d <= 0:
        raise ValueError("l and d must be positive integers.")
    m = 1 << (d * l)   # 2^(d*l)
    n = d
    N = m * n

    cols = np.arange(N, dtype=np.int64)             # input basis indices
    s = cols % n                                    # s = input index mod n
    j = cols // n                                   # j = input index div n
    rows = s * m + j                                # output basis indices

    if sparse:
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("scipy is required for sparse output; install scipy or set sparse=False.") from e
        data = np.ones(N, dtype=dtype)
        return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=dtype)
    else:
        P = np.zeros((N, N), dtype=dtype)
        P[rows, cols] = 1
        return np.kron(P,np.eye(2**d))

# return the product of T_{m,m+1,1D} matrices where m ranges from l to L-1
def get_T_1D(l: int, L: int):
    if l == L:
        return np.eye(2**(l+1))
    elif l == L-1:
        Z = np.array([[1,0],[0,-1]])
        '''Z_kron = Z
        for i in range(l-1):
            Z_kron = np.kron(Z_kron, Z)
        return (1/math.sqrt(2)) * np.kron(Z_kron,np.array([[1, -math.sqrt(3)/2],[0, 1/2],[1, math.sqrt(3)/2],[0, 1/2]]))'''
        #return (1/math.sqrt(2)) * np.kron(np.eye(2**(l-1)), np.kron(np.array([[1,0],[0,-1]]),np.array([[1, -math.sqrt(3)/2],[0, 1/2],[1, math.sqrt(3)/2],[0, 1/2]])))
        return (1/math.sqrt(2)) * np.kron(np.eye(2**l), np.array([[1, -math.sqrt(3)/2],[0, 1/2],[1, math.sqrt(3)/2],[0, 1/2]]))
    else:
        return get_T_1D(L-1,L) @ get_T_1D(l,L-1)
    

# the domain goes from 0 to 1
D = 2
L = 3 # number of levels of BPX preconditioner
n_fine = int(math.pow(2,L)) # number of points in finest level
h_fine = 1/n_fine
CF_col_sections = [(2**l-1)**D for l in range(1,L+1)] # list of number of basis functions in each level

# create the 1-D F matrix preconditioner (defined at the bottom of page 11 of the Deiml paper)
F = np.zeros(((n_fine-1)**D, sum(CF_col_sections)))
fine_position_vals = [np.linspace(h_fine,1-h_fine,n_fine-1) for i in range(D)]
position_meshes = np.meshgrid(*fine_position_vals, indexing='ij')
position_points = [position_meshes[i].flatten() for i in range(D)]
col = 0
for l in range(1,L+1):
    n_coarse = int(math.pow(2,l)) # number of basis functions in current level
    h_coarse = 1/n_coarse
    #triangle_height_weight = math.pow(2,l-L)
    triangle_height_weight = 1 # height of hat function in basis function
    for i in range((n_coarse - 1)**D): # find values of current column of F
        pos_indices = [int(i%(n_coarse-1)**(D-d) / (n_coarse-1)**(D-d-1)) for d in range(D)]
        pos = [position_points[d][i] for d in range(D)]
        weights_list = [[2 ** (-l * (2-D) / 2) * triangle_height_weight * triangle_wave(x,h_coarse*pos_indices[d],h_coarse*(pos_indices[d]+2)) for x in fine_position_vals[d]] for d in range(D)]
        weights = weights_list[0]
        for d in range(1,D):
            weights = np.kron(weights, weights_list[d])
        F[:,col] =  weights
        col += 1
#print("F condition number:", np.linalg.cond(F))
#print("F matrix norm: ", np.linalg.norm(F))
#print(F)

# find the 1 dimensional C_l matrix using the definition in Theorem 6.3 of the Deiml paper
#l = 0
#C_l_1D = get_C_l_1D(l)
#R_l_1D = get_R_l_1D(l)
#print(C_l_1D)

# find C_L (C_l for the finest level)
pi_l_C_L = np.zeros((D*2**(D*(L+1)), (2**L - 1)**D))
for s in range(1,D+1):
    pi_l_C_L_s = np.array([1])
    for i in range(1,s):
        pi_l_C_L_s = np.kron(pi_l_C_L_s, get_R_l_1D(L))
    pi_l_C_L_s = np.kron(pi_l_C_L_s, get_C_l_1D(L))
    for i in range(s+1, D+1):
        pi_l_C_L_s = np.kron(pi_l_C_L_s, get_R_l_1D(L))
    pi_l_C_L[(s-1)*2**(D*(L+1)):s*2**(D*(L+1)), :] = pi_l_C_L_s
pi_L_star = jk_interleave_permutation_matrix(L, D, sparse=False)
pi_L = np.kron(np.eye(D), pi_L_star)
C_L = np.transpose(pi_L) @ pi_l_C_L

C_F_test = C_L @ F


pi_L_star = jk_interleave_permutation_matrix(L, D, sparse=False) # permutation matrix of one of D blocks of C_l 
pi_L = np.kron(np.eye(D), pi_L_star) #permutation matrix of entire C_l matrix
CF = np.zeros((D * 2**(D*(L+1)), sum(CF_col_sections)))
for l in range(1,L+1):
    pi_l_C_l = np.zeros((D*2**(D*(l+1)), (2**l - 1)**D))
    for s in range(1,D+1):
        pi_l_C_l_s = np.array([1])
        for i in range(1,s):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D(l))
        pi_l_C_l_s = np.kron(pi_l_C_l_s, get_C_l_1D(l))
        for i in range(s+1, D+1):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D(l))
        pi_l_C_l[(s-1)*2**(D*(l+1)):s*2**(D*(l+1)), :] = pi_l_C_l_s
    pi_l_star = jk_interleave_permutation_matrix(l, D, sparse=False)
    pi_l = np.kron(np.eye(D), pi_l_star)
    C_l = np.transpose(pi_l) @ pi_l_C_l

    T_1D = get_T_1D(l,L)
    T = np.array([1])
    for i in range(D):
        T = np.kron(T_1D, T) # kronecker product the T_1D matrix product D times (bottom of page 16 of Deiml paper)
    T = np.transpose(pi_L_star) @ T @ pi_l_star # apply the permutations on  T

    pi_left = js_swap_perm_matrix(L, D, sparse=False) # different permutation matrix than pi_l that switches the j and s registers
    pi_right = js_swap_perm_matrix(l, D, sparse=False)
    T_squiggle = np.transpose(pi_left) @ np.kron(np.eye(D), T) @ pi_right # top of page 16 of the Deiml paper
    CFl = 2 ** (-l * (2-D) / 2) * T_squiggle @ C_l # section s corresponding to level l of the CF matrix
    #CFl = T_squiggle @ C_l # section s corresponding to level l of the CF matrix

    CF[:,sum(CF_col_sections[:l-1]):sum(CF_col_sections[:l])] = CFl

#mat_L = 2
#diffusion_mat_small = np.diag(np.random.rand(2**(D*mat_L))) # matrix must be 2^(D*mat_L) X 2^(D*mat_L)
#diffusion_mat = np.kron(diffusion_mat_small, np.eye(2**(L - mat_L)))
#diffusion_mat = np.diag(np.random.rand(2**(D*L))) # matrix must be 2^(D*L) X 2^(D*L)
#diffusion_mat = np.diag(3*np.ones(2**(D*L)))
diffusion_mat = np.diag(np.concatenate((3*np.ones(2**(D*L-1)),5*np.ones(2**(D*L-1))))) # matrix of diffusion coefficients
D_A = np.kron(diffusion_mat, np.eye(D))


S = np.transpose(C_l) @ np.kron(D_A, np.eye(2**D)) @ C_l

# test consistency of solution
F_remainder = (np.eye(D*2**(D*(l+1))) - C_l @ np.linalg.pinv(C_l)) @ CF # has to be equal to 0 for a solution to exist
print("Max F-remainder: ", np.max(abs(F_remainder)))

# The CF here (which is the C_l found using the equations in the Deiml paper (page 15) multiplied by the F preconditioner (page 12)) 
# is not the same as the CF we get from making the full matrix using the steps from the same Deiml paper however, this C_F_test is very
# close to CF, and both I think can be implemented in logN time, so whichever one is right should be still be able to be encoded quickly
#C_F_test = C_l @ F
#CF = C_F_test

CF_first_column_sum = np.max(CF[:,0])
CF_test_first_column_sum = np.max(C_F_test[:,0])

F_test = np.linalg.pinv(C_l) @ CF # The preconditioner F matrix if we assume that we created CF correctly
CF_test2 = C_l @ F_test
FSF1 = np.transpose(F) @ S @ F # preconditioned system using the F matrix
FSF2 = np.transpose(CF) @ np.kron(D_A, np.eye(2**D)) @ CF # preconditioned system using the CF matrix (should be the same as FSF1)
FSF_error_mat = FSF1 - FSF2
FSF_error = np.linalg.norm(FSF_error_mat)
print("FSF error: ", FSF_error)

print("done")



