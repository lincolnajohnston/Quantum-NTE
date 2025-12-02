import numpy as np
import math
import scipy as sp
from scipy.sparse import csr_matrix, coo_matrix
import itertools

# Some helper functions to create the matrices used for the BPX preconditioner and for the FEM discretization of the diffusion equation 


# return the y value (as fraction of max height) at position x for a triangle wave starting at x_min and ending at x_max
def triangle_wave(x: float, x_min, x_max) -> float:
    if x < x_min or x > x_max:
        return 0
    dx = x_max - x_min
    r = x - x_min
    return 1.0 - 2.0 * abs(r - dx/2) / dx

def get_C_l_1D(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[1,-1],[0,0]])) # TODO: figure out how this works, don't really understand this equation
    # M2 from the Deiml paper
    #I_l = np.eye(N)[:,:N-1]
    #N_l = np.eye(N)[:,1:N]
    #M2 = np.concatenate((I_l, N_l), axis=0)

    M2 = np.zeros((2**(l+1), 2**l - 1))
    # Lincoln's M2, now an operation performing |i> -> 1/sqrt(2) (|2i+1> + |2i+2>)
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
    M1 = np.kron(np.eye(N), np.array([[1/2,1/2],[-1/(2*math.sqrt(3)),1/(2*math.sqrt(3))]]))

    # Deiml paper's M2
    #I_l = np.eye(N)[:,:N-1]
    #N_l = np.eye(N)[:,1:N]
    #M2 = np.concatenate((I_l, N_l), axis=0)

    # Lincoln's M2
    M2 = np.zeros((2**(l+1), 2**l - 1))
    for col in range(2**l - 1):
        M2[2*col+1, col] = 1
        M2[2*(col+1), col] = 1
    return 2**(-l/2) * M1 @ M2

# ChatGPT function, changed the implementation now, not completely checked for correctness
# return the Pi_l operator in the middle of page 15 of the Deiml paper
def jk_interleave_permutation_matrix(l: int, d: int, sparse: bool = True, dtype=np.uint8, inverse: bool=False):
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
    J_starts = [i * (l + int(inverse)) for i in range(d)]
    if inverse:
        for i in range(d):
            # J_i block of length l
            mapping_left.extend(range(J_starts[i], J_starts[i] + l))
        for i in range(d):
            # K_i is a single bit at left-position d*l + i in the input
            mapping_left.append(i * (l + 1) + l)
    else:
        for i in range(d):
            # J_i block of length l
            mapping_left.extend(range(J_starts[i], J_starts[i] + l))
            # K_i is a single bit at left-position d*l + i in the input
            mapping_left.append(d * l + i)
    

    mapping_left = np.asarray(mapping_left, dtype=np.int64)  # length n

    # Convert left-to-right positions to bit indices (0 = LSB)
    dest_bit = mapping_left          # output bit indices (LSB-first)

    # Build permutation of basis indices: for each input column x -> output row y
    rows = np.empty(N, dtype=np.int64)
    cols = np.arange(N, dtype=np.int64)
    for x in range(N):
        x_bin = bin(x)[2:].zfill(n)
        y_bin = [x_bin[dest_bit[i]] for i in range(n)]
        y = int("".join(y_bin), 2)
        rows[x] = y
    return rows

def getC_l(D, l):
    pi_l_C_l = csr_matrix((D*2**(D*(l+1)), (2**l - 1)**D), dtype=float)
    for s in range(1,D+1):
        pi_l_C_l_s = np.array([1])
        for _ in range(1,s):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D(l))
        pi_l_C_l_s = np.kron(pi_l_C_l_s, get_C_l_1D(l))
        for _ in range(s+1, D+1):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D(l))
        rows, cols = np.nonzero(pi_l_C_l_s)
        pi_l_C_l_s_data = pi_l_C_l_s[rows, cols]

        rows_g = rows + (s-1)*2**(D*(l+1))
        cols_g = cols

        update = coo_matrix((pi_l_C_l_s_data, (rows_g, cols_g)), shape=pi_l_C_l.shape)
        pi_l_C_l += update.tocsr() 
        
        #pi_l_C_l[(s-1)*2**(D*(l+1)):s*2**(D*(l+1)), :] = pi_l_C_l_s
    pi_l_star_new = jk_interleave_permutation_matrix(l, D, sparse=False)
    pi_l_new = np.array([pi_l_star_new + 2**(D*l+D) * d for d in range(D)]).flatten()
    C_l = pi_l_C_l[pi_l_new, :]
    return C_l

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

# return the product of T_{m,m+1,1D} matrices where m ranges from l to L-1
# defined at the bottom of page 16 of the Deiml paper
def get_T_1D(l: int, L: int):
    if l == L:
        return np.eye(2**(l+1))
    elif l == L-1:
        return (1/math.sqrt(2)) * np.kron(np.eye(2**l), np.array([[1, -math.sqrt(3)/2],[0, 1/2],[1, math.sqrt(3)/2],[0, 1/2]]))
    else:
        return get_T_1D(L-1,L) @ get_T_1D(l,L-1)
    
# T_squiggle defined at the top of page 14 of the Deiml paper
def get_T_squiggle(D, l, L):
    pi_l_star = jk_interleave_permutation_matrix(l, D, sparse=False)
    pi_L_star = jk_interleave_permutation_matrix(L, D, sparse=False)
    #pi_L = np.array([pi_L_star + 2**(D*L+D) * d for d in range(D)]).flatten()

    T_1D = sp.sparse.csr_array(get_T_1D(l,L))
    T = sp.sparse.csr_array([1])
    for i in range(D):
        #T = np.kron(T_1D, T) # kronecker product the T_1D matrix product D times (bottom of page 16 of Deiml paper)
        T = sp.sparse.kron(T_1D, T, format="csr")
    T = T[:,pi_l_star] # apply column permutation (right of T)
    T = T[pi_L_star,:] # apply row permutation (left of T)

    # top of page 16 of the Deiml paper
    T_squiggle = sp.sparse.kron(np.eye(D), T, format="csr")
    return T_squiggle
