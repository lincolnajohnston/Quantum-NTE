import numpy as np
import math
import scipy as sp
import scipy.sparse as spsp
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
    M1 = np.kron(np.eye(N), np.array([[-1,1],[0,0]])) # slightly modified from the Deiml paper (switched the 1 and -1), I don't think it majorly affects anything but aligns with my math better
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
    test = 2**(l/2) * M1 @ M2
    return 2**(l/2) * M1 @ M2

# return the R_{l,1D} matrix at the bottom of page 15 of the Deiml paper 
def get_R_l_1D(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[1/2,1/2],[-1/(2*math.sqrt(3)),1/(2*math.sqrt(3))]])) # modified from the Deiml paper, but I think this is correct (Deiml paper might just have a different ordering of basis functions?)

    # Deiml paper's M2
    #I_l = np.eye(N)[:,:N-1]
    #N_l = np.eye(N)[:,1:N]
    #M2 = np.concatenate((I_l, N_l), axis=0)

    # Lincoln's M2
    M2 = np.zeros((2**(l+1), 2**l - 1))
    for col in range(2**l - 1):
        M2[2*col+1, col] = 1
        M2[2*(col+1), col] = 1
    test = 2**(-l/2) * M1 @ M2
    return 2**(-l/2) * M1 @ M2

# C_l_1D for the vacuum boundary condition FEM space
def get_C_l_1D_v(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[-1,1],[0,0]])) # slightly modified from the Deiml paper (switched the 1 and -1), I don't think it majorly affects anything but aligns with my math better

    M2 = np.zeros((2**(l+1), 2**l + 1))
    # Modified M2 for vacuum BC, now an operation performing |i> -> 1/sqrt(2) (|2i> + |2i-1>) for 0 < i < 2**l and |0> -> 1/sqrt(2) |0> and |2**l> -> 1/sqrt(2)|2**(l+1)-1>
    for col in range(1,2**l):
        M2[2*col, col] = 1
        M2[2*col-1, col] = 1
    M2[0, 0] = 1
    M2[2**(l+1)-1, 2**l] = 1
    test = 2**(l/2) * M1 @ M2
    return 2**(l/2) * M1 @ M2

def get_R_l_1D_v(l):
    N = 2 ** l
    M1 = np.kron(np.eye(N), np.array([[1/2,1/2],[-1/(2*math.sqrt(3)),1/(2*math.sqrt(3))]])) # modified from the Deiml paper, but I think this is correct (Deiml paper might just have a different ordering of basis functions?)

    M2 = np.zeros((2**(l+1), 2**l + 1))
    # Modified M2 for vacuum BC, now an operation performing |i> -> 1/sqrt(2) (|2i> + |2i-1>) for 0 < i < 2**l and |0> -> 1/sqrt(2) |0> and |2**l> -> 1/sqrt(2)|2**(l+1)-1>
    for col in range(1,2**l):
        M2[2*col+1, col] = 1
        M2[2*(col+1), col] = 1
    M2[0] = 0
    M2[2**(l+1)-1] = 2**l

    test = 2**(-l/2) * M1 @ M2
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

# get C_l for vacuum boundary condition
def getC_l_v(D, l):
    pi_l_C_l = csr_matrix((D*2**(D*(l+1)), (2**l + 1)**D), dtype=float)
    for s in range(1,D+1):
        pi_l_C_l_s = np.array([1])
        for _ in range(1,s):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D_v(l))
        pi_l_C_l_s = np.kron(pi_l_C_l_s, get_C_l_1D_v(l))
        for _ in range(s+1, D+1):
            pi_l_C_l_s = np.kron(pi_l_C_l_s, get_R_l_1D_v(l))
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

# get the basis change matrix from mutlilevel basis to the basis of the finest level with vacuum boundary conditions
def get_F_v(D, L):
    n_fine = int(math.pow(2,L)) # number of points in finest level
    h_fine = 1/n_fine
    CF_col_sections = [(2**l+1)**D for l in range(1,L+1)] # list of number of basis functions in each level

    # create the 1-D F matrix preconditioner (defined at the bottom of page 11 of the Deiml paper)
    F = csr_matrix(((n_fine+1)**D, sum(CF_col_sections)), dtype=float)
    fine_position_vals = [np.linspace(0,1,n_fine+1) for i in range(D)]
    position_meshes = np.meshgrid(*fine_position_vals, indexing='ij')
    position_points = [position_meshes[i].flatten() for i in range(D)]
    col = 0
    for l in range(1,L+1):
        n_coarse = int(math.pow(2,l)) # number of basis functions in current level
        h_coarse = 1/n_coarse
        triangle_height_weight = 1 # height of hat function in basis function
        level_weight = 2 ** (-l * (2-D) / 2)
        #level_weight = 1
        for i in range((n_coarse + 1)**D): # find values of current column of F
            pos_indices = [int(i%(n_coarse+1)**(D-d) / (n_coarse+1)**(D-d-1)) for d in range(D)]
            pos = [position_points[d][i] for d in range(D)]
            weights_list = np.array([[triangle_height_weight * triangle_wave(x,h_coarse*(pos_indices[d]-1),h_coarse*(pos_indices[d]+1)) for x in fine_position_vals[d]] for d in range(D)])
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

# given dimensions D, levels L, and a matrix of diffusion coefficients (which must be able to be divided 
# evenly onto the FEM grid discretization), return the FEM diffusion matrix
def get_diffusion_matrix(D, L, dif_mat):
    #mat_L = int(np.log2(len(diffusion_mat_small)) / D) # level of the material discretization for defining diffusion coefficients
    # ensure that the length of diffusion mat is equal to a power of 2
    #if np.abs(mat_L % 1) > 0.00000001:
    #    return 0

    D_A = np.kron(dif_mat, np.eye(D))
    C_L = getC_l(D, L)
    A = C_L.T @ spsp.kron(D_A, spsp.eye(2**D, format="csr"), format="csr") @ C_L
    return A.tocsr()

# convert multidimensional (x1,x2...,xD) index to 1D index, first index is most significant
def unroll_index(N, D, index_vec, xs_mesh=False):
    roll_N = [N]*D
    roll_N = roll_N + 1 if xs_mesh else roll_N
    return sum([index_vec[d]*math.prod(roll_N[d+1:]) for d in range(D)])

 # assume domain is 1 so h = 1/2^L, factor out the h^D term

'''def get_mass_matrix_brute_force(L, D, xs):
    N_1D_FEM = int(2**(L) - 1)
    N_total = int(N_1D_FEM**D)
    M = np.zeros((N_total, N_total))

    all_indices = itertools.product(list(range(N_1D_FEM)), repeat=D)
    for node_index in all_indices: # iterate through all nodes
        offset_indices = itertools.product(list(range(-1,2)), repeat=D)
        for node_offset_index in offset_indices: # iterate through each node surrounding the current node
                row_index = np.array(node_index)
                col_index = row_index + np.array(node_offset_index)

                # skip the nodes that are outside the domain
                valid_index = True
                for d in range(D):
                    if col_index[d] < 0 or col_index[d] >= N_1D_FEM:
                        valid_index = False
                if not valid_index:
                    continue

                xs_coef = 2**(D-sum(np.abs(np.array(node_offset_index)))) / 6**D # coefficient on each of the cross sections in the matrix
                sigma_index_lower = row_index + [max(offset, 0) for offset in node_offset_index] # lower index of the xs terms
                sigma_index_upper = [min(col_index[i], row_index[i]) + 1 for i in range(len(row_index))] # upper index of the xs terms
                xs_indices_list = [list(range(sigma_index_lower[d], sigma_index_upper[d]+1)) for d in range(D)]
                xs_indices = itertools.product(*xs_indices_list)
                for xs_index in xs_indices:
                    M[unroll_index(N_1D_FEM, D, row_index), unroll_index(N_1D_FEM, D, col_index)] += xs_coef * xs[xs_index]
    return M '''  

# get the mass matrix for the absorption or fission matrix (without the h^D factor in front)
def get_mass_matrix_brute_force(L, D, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    rows = []
    cols = []
    data = []

    all_indices = itertools.product(range(N_1D_FEM), repeat=D)

    for node_index in all_indices:
        node_index = np.array(node_index)

        for offset in itertools.product(range(-1,2), repeat=D):
            col_index = node_index + np.array(offset)

            # skip outside domain
            if np.any(col_index < 0) or np.any(col_index >= N_1D_FEM):
                continue

            xs_coef = 2**(D - sum(abs(o) for o in offset)) / 6**D # get the coefficient in front of the matrix term

            sigma_index_lower = node_index + [max(o,0) for o in offset]
            sigma_index_upper = [min(col_index[i],node_index[i])+1 for i in range(D)]

            xs_ranges = [range(sigma_index_lower[d], sigma_index_upper[d]+1) for d in range(D)]

            val = 0.0
            for xs_index in itertools.product(*xs_ranges):
                val += xs_coef * xs[xs_index]

            i = unroll_index(N_1D_FEM, D, node_index)
            j = unroll_index(N_1D_FEM, D, col_index)

            rows.append(i)
            cols.append(j)
            data.append(val)

    return coo_matrix((data,(rows,cols)),shape=(N_total,N_total)).tocsr()

def get_2D_diffusion_matrix_brute_force(L, D, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    rows = []
    cols = []
    data = []

    all_indices = itertools.product(range(N_1D_FEM), repeat=D)

    for node_index in all_indices:
        node_index = np.array(node_index)

        for offset in itertools.product(range(-1,2), repeat=D):
            col_index = node_index + np.array(offset)

            # skip outside domain
            if np.any(col_index < 0) or np.any(col_index >= N_1D_FEM):
                continue
            

            xs_coef = -1/6
            if offset[0]==0 and offset[1]==0:
                xs_coef = 2/3
            if abs(offset[0])==1 and abs(offset[1])==1:
                xs_coef = -1/3
            

            sigma_index_lower = node_index + [max(o,0) for o in offset]
            sigma_index_upper = [min(col_index[i],node_index[i])+1 for i in range(D)]

            xs_ranges = [range(sigma_index_lower[d], sigma_index_upper[d]+1) for d in range(D)]

            val = 0.0
            for xs_index in itertools.product(*xs_ranges):
                val += xs_coef * xs[xs_index]

            i = unroll_index(N_1D_FEM, D, node_index)
            j = unroll_index(N_1D_FEM, D, col_index)

            rows.append(i)
            cols.append(j)
            data.append(val)

    return coo_matrix((data,(rows,cols)),shape=(N_total,N_total)).tocsr()


# return a FEM mass matrix where each term is weighted by a piecewise-constant value (like the absorption cross section)
# L is number of levels, consts is an array of constants defined in each cell of the FEM discretization
def get_M1(L, consts):
    # make sure consts is the right size
    if len(consts) != int(2**(L)):
        raise ValueError("consts matrix is not the correct size/shape")
    h = 1/(2**L)
    A = np.diag(2*consts[:-1])
    A += np.diag(2*consts[1:])
    A += np.diag(consts[1:-1], k=1)
    A += np.diag(consts[1:-1], k=-1)
    return (h/6)*A # weight the matrix by a scalar

def get_P1(L, consts):
    # make sure consts is the right size
    if len(consts) != int(2**(L)):
        raise ValueError("consts matrix is not the correct size/shape")
    h = 1/(2**L)
    A = np.diag(consts[:-1])
    A += np.diag(consts[1:])
    A += np.diag(-1*consts[1:-1], k=1)
    A += np.diag(-1*consts[1:-1], k=-1)
    return (1/h)*A # weight the matrix by a scalar



# given a cross section vector, xs, and a list of index ranges ([[x_low, x_high], [y_low, y_high], ...])
# return the vector of cross sections only within those index ranges
def get_xs_subvector(D, xs_list, index_ranges):
    return np.array(xs_list[np.ix_(*[list(range(index_ranges[d][0], index_ranges[d][1]+1)) for d in range(D)])]).flatten()


def get_1D_mass_matrix_LCU(D, L, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    diagonals = [] # list of diagonal arrays
    offsets = [] # list of offsets corresponding to the diagonals

    ########## main diagonal #########
    coef = (1/3)**D # each overlapping hat function's integral product is 1/3
    all_indices = itertools.product(list(range(2)), repeat=D) # starting indices for xs list
    main_diagonal = np.zeros(N_1D_FEM)
    for start_indices in all_indices:
        main_diagonal += coef * get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D)]) # get cross sections for all but one index in each dimension (0...N-1 or 1...N)
    diagonals.append(main_diagonal)
    offsets.append(0)

    ######### 1 offset hat function in x direction #########
    coef = (1/3)**(D-1) * (1/6)**1 # offset hat functions have a square integral of 1/6
    # LSB (least significant bit) increment, for D > 0
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the more significant bits
    diag_len = N_1D_FEM - 1
    x_offset_diag = np.zeros(diag_len)
    for start_indices in all_indices:
        diag = get_xs_subvector(D, xs, [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)

        diag_zero_inserts = diag
        x_offset_diag += coef * diag_zero_inserts # above the main diagonal
    diagonals.append(x_offset_diag)
    diagonals.append(x_offset_diag)
    offsets.append(1)
    offsets.append(-1)

    return spsp.diags(diagonals, offsets)

# does not include 1/h factor in front
def get_1D_diffusion_matrix_LCU(D, L, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    diagonals = [] # list of diagonal arrays
    offsets = [] # list of offsets corresponding to the diagonals

    ########## main diagonal #########
    coef = 1 # each overlapping hat function's integral product is 1/3
    all_indices = itertools.product(list(range(2)), repeat=D) # starting indices for xs list
    main_diagonal = np.zeros(N_1D_FEM)
    for start_indices in all_indices:
        main_diagonal += coef * get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D)]) # get cross sections for all but one index in each dimension (0...N-1 or 1...N)
    diagonals.append(main_diagonal)
    offsets.append(0)

    ######### 1 offset hat function in x direction #########
    coef = -1 # offset hat functions have a square integral of 1/6
    # LSB (least significant bit) increment, for D > 0
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the more significant bits
    diag_len = N_1D_FEM - 1
    x_offset_diag = np.zeros(diag_len)
    for start_indices in all_indices:
        diag = get_xs_subvector(D, xs, [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)

        diag_zero_inserts = diag
        x_offset_diag += coef * diag_zero_inserts # above the main diagonal
    diagonals.append(x_offset_diag)
    diagonals.append(x_offset_diag)
    offsets.append(1)
    offsets.append(-1)

    return spsp.diags(diagonals, offsets)

# assume domain is 1 so h = 1/2^L, factor out the h^D term
# craft the matrix by taking the linear combination of diagonal matrices, each of which 
# can be efficiently block-encoded and then ocmbined with LCU.
# This function is uncompleted, just here to show that the fission and absorption FEM
# matrices can be implemented as the linear combination of 6^D diagonal matrices (with some integer shift |x> -> |x+1>)
def get_2D_mass_matrix_LCU(D, L, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    diagonals = [] # list of diagonal arrays
    offsets = [] # list of offsets corresponding to the diagonals

    ########## main diagonal #########
    coef = (1/3)**D # each overlapping hat function's integral product is 1/3
    all_indices = itertools.product(list(range(2)), repeat=D) # starting indices for xs list
    main_diagonal = np.zeros(N_1D_FEM*N_1D_FEM)
    for start_indices in all_indices:
        main_diagonal += coef * get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D)]) # get cross sections for all but one index in each dimension (0...N-1 or 1...N)
        #M += coef * np.diag(diag) # add the cross sections to the main diagonal
    diagonals.append(main_diagonal)
    offsets.append(0)

    ######### 1 offset hat function in x direction #########
    coef = (1/3)**(D-1) * (1/6)**1 # offset hat functions have a square integral of 1/6
    # LSB (least significant bit) increment, for D > 0
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the more significant bits
    diag_len = N_1D_FEM**2 - 1
    x_offset_diag = np.zeros(diag_len)
    for start_indices in all_indices:
        diag = get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D-1)] + [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)
        # need to add in the zeros between the sections here
        diag_zero_inserts = np.zeros(diag_len)
        mask = np.ones(diag_len, dtype=bool)
        mask[N_1D_FEM-1::(N_1D_FEM)] = False   # every (n)th position is a zero

        diag_zero_inserts[mask] = diag
        x_offset_diag += coef * diag_zero_inserts # above the main diagonal
    diagonals.append(x_offset_diag)
    diagonals.append(x_offset_diag)
    offsets.append(1)
    offsets.append(-1)

    ######### 1 offset hat function in y direction #########
    coef = (1/3)**(D-1) * (1/6)**1 # offset hat functions have a square integral of 1/6
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the less significant bits
    y_offset_diag = np.zeros((N_1D_FEM-1)*N_1D_FEM)
    for start_indices in all_indices:
        y_offset_diag += coef * get_xs_subvector(D, xs, [[1,N_1D_FEM-1]] + [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D-1)]) # get the diagonal values (except for zero values)

        #M += coef * np.diag(diag, k=N_1D_FEM) # above the main diagonal
        #M += coef * np.diag(diag, k=-N_1D_FEM) # under the main diagonal

    diagonals.append(y_offset_diag)
    diagonals.append(y_offset_diag)
    offsets.append(N_1D_FEM)
    offsets.append(-N_1D_FEM)


    ######## 1 offset hat function in x and y direction ########
    coef =  (1/6)**2 # offset hat functions have a square integral of 1/6
    diag = get_xs_subvector(D, xs, [[1,N_1D_FEM-1]] + [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)

    diag_len_ext = (N_1D_FEM-1) * (N_1D_FEM) - 1
    diag_len_int = (N_1D_FEM-1) * (N_1D_FEM) + 1
    # need to add in the zeros between the sections here
    diag_zero_inserts_int = np.zeros(diag_len_int)
    diag_zero_inserts_ext = np.zeros(diag_len_ext)
    mask_interior = np.ones(diag_len_int, dtype=bool)
    mask_interior[0::(N_1D_FEM)] = False   # every (n-1)th position is a zero
    mask_exterior = np.ones(diag_len_ext, dtype=bool)
    mask_exterior[N_1D_FEM-1::(N_1D_FEM)] = False   # every (n-1)th position is a zero

    diag_zero_inserts_int[mask_interior] = diag
    diag_zero_inserts_ext[mask_exterior] = diag

    # interior diagonals
    diagonals.append(coef*diag_zero_inserts_int)
    diagonals.append(coef*diag_zero_inserts_int)
    offsets.append(N_1D_FEM-1)
    offsets.append(-N_1D_FEM+1)

    # exterior diagonals
    diagonals.append(coef*diag_zero_inserts_ext)
    diagonals.append(coef*diag_zero_inserts_ext)
    offsets.append(N_1D_FEM+1)
    offsets.append(-N_1D_FEM-1)

    return spsp.diags(diagonals, offsets)



def get_2D_diffusion_matrix_LCU(D, L, xs):
    N_1D_FEM = 2**L - 1
    N_total = N_1D_FEM**D

    diagonals = [] # list of diagonal arrays
    offsets = [] # list of offsets corresponding to the diagonals

    ########## main diagonal #########
    coef = (2/3) # each overlapping hat function's integral product is 1/3
    all_indices = itertools.product(list(range(2)), repeat=D) # starting indices for xs list
    main_diagonal = np.zeros(N_1D_FEM*N_1D_FEM)
    for start_indices in all_indices:
        main_diagonal += coef * get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D)]) # get cross sections for all but one index in each dimension (0...N-1 or 1...N)
        #M += coef * np.diag(diag) # add the cross sections to the main diagonal
    diagonals.append(main_diagonal)
    offsets.append(0)

    ######### 1 offset hat function in x direction #########
    coef = -1/6 # offset hat functions have a square integral of 1/6
    # LSB (least significant bit) increment, for D > 0
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the more significant bits
    diag_len = N_1D_FEM**2 - 1
    x_offset_diag = np.zeros(diag_len)
    for start_indices in all_indices:
        diag = get_xs_subvector(D, xs, [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D-1)] + [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)
        # need to add in the zeros between the sections here
        diag_zero_inserts = np.zeros(diag_len)
        mask = np.ones(diag_len, dtype=bool)
        mask[N_1D_FEM-1::(N_1D_FEM)] = False   # every (n)th position is a zero

        diag_zero_inserts[mask] = diag
        x_offset_diag += coef * diag_zero_inserts # above the main diagonal
    diagonals.append(x_offset_diag)
    diagonals.append(x_offset_diag)
    offsets.append(1)
    offsets.append(-1)

    ######### 1 offset hat function in y direction #########
    coef = -1/6 # offset hat functions have a square integral of 1/6
    all_indices = itertools.product(list(range(2)), repeat=D-1) # just the index range for the less significant bits
    y_offset_diag = np.zeros((N_1D_FEM-1)*N_1D_FEM)
    for start_indices in all_indices:
        y_offset_diag += coef * get_xs_subvector(D, xs, [[1,N_1D_FEM-1]] + [[start_indices[d], N_1D_FEM-1+start_indices[d]] for d in range(D-1)]) # get the diagonal values (except for zero values)

        #M += coef * np.diag(diag, k=N_1D_FEM) # above the main diagonal
        #M += coef * np.diag(diag, k=-N_1D_FEM) # under the main diagonal

    diagonals.append(y_offset_diag)
    diagonals.append(y_offset_diag)
    offsets.append(N_1D_FEM)
    offsets.append(-N_1D_FEM)


    ######## 1 offset hat function in x and y direction ########
    coef =  -1/3# offset hat functions have a square integral of 1/6
    diag = get_xs_subvector(D, xs, [[1,N_1D_FEM-1]] + [[1,N_1D_FEM-1]]) # get the diagonal values (except for zero values)

    diag_len_ext = (N_1D_FEM-1) * (N_1D_FEM) - 1
    diag_len_int = (N_1D_FEM-1) * (N_1D_FEM) + 1
    # need to add in the zeros between the sections here
    diag_zero_inserts_int = np.zeros(diag_len_int)
    diag_zero_inserts_ext = np.zeros(diag_len_ext)
    mask_interior = np.ones(diag_len_int, dtype=bool)
    mask_interior[0::(N_1D_FEM)] = False   # every (n-1)th position is a zero
    mask_exterior = np.ones(diag_len_ext, dtype=bool)
    mask_exterior[N_1D_FEM-1::(N_1D_FEM)] = False   # every (n-1)th position is a zero

    diag_zero_inserts_int[mask_interior] = diag
    diag_zero_inserts_ext[mask_exterior] = diag

    # interior diagonals
    diagonals.append(coef*diag_zero_inserts_int)
    diagonals.append(coef*diag_zero_inserts_int)
    offsets.append(N_1D_FEM-1)
    offsets.append(-N_1D_FEM+1)

    # exterior diagonals
    diagonals.append(coef*diag_zero_inserts_ext)
    diagonals.append(coef*diag_zero_inserts_ext)
    offsets.append(N_1D_FEM+1)
    offsets.append(-N_1D_FEM-1)

    return spsp.diags(diagonals, offsets)


# return the mass matrix for a D-dimensional domain with 2^(L_f) node in each dimension
# xs is the D-dimensional (2^L_m x 2^L_m x ... ) numpy matrix representing the material
# discretization of the problem 
def get_mass_matrix_LCU(D, L_f, mat_L, xs):
    N_f = int(2**L_f)
    xs_fine = np.kron(xs, np.ones((int(2**(L_f-mat_L)),) * D))
    mass_matrix = np.zeros(((N_f-1)**D, (N_f-1)**D))
    for offsets in itertools.product([-1,0,1], repeat=D):
        idx_starts = [[0,1]]*D # start indices for the submatrices to create the diagonals of the mass matrix
        xs_fine_temp = xs_fine.copy()
        for d, offset in enumerate(offsets): # d=0 is most significant 
            if offset == -1:
                xs_fine_temp = xs_fine_temp[(slice(None),)*d + (slice(0, N_f-1),)] # leaves the first d dimensions unchanged, removes the last index of dimension d
                xs_fine_temp[(slice(None),)*d + (slice(0, 1),) + (slice(None),)*(D-d-1)] = 0 # leaves the first d dimensions unchanged, sets index 0 of dimension d to 0, leaves the remaining D-d-1 dimensions unchanged
                idx_starts[d] = [0]
            if offset == 1:
                xs_fine_temp = xs_fine_temp[(slice(None),)*d + (slice(1, None),)] # leaves the first d dimensions unchanged, removes the first index of dimension d
                xs_fine_temp[(slice(None),)*d + (slice(N_f-2, N_f-1),) + (slice(None),)*(D-d-1)] = 0 # leaves the first d dimensions unchanged, sets the last index of dimension d to 0, leaves the remaining D-d-1 dimensions unchanged
                idx_starts[d] = [0]
        diag_vec = np.zeros((N_f-1)**D)
        for idx_start in itertools.product(*idx_starts):
            temp_vec = xs_fine_temp[tuple(slice(idx_start[d], idx_start[d] + (N_f - 1)) for d in range(D))]
            diag_vec += temp_vec.flatten()
        diag_mat = np.diag(diag_vec)
        offset_magnitudes = np.array([(N_f-1)**d for d in range(D-1,-1,-1)])
        offset = sum(np.array(offsets) * offset_magnitudes)
        diag_mat = np.roll(diag_mat, shift=offset, axis=0)
        mass_matrix += diag_mat
    return mass_matrix


# do the same thing as get_mass_matrix_LCU except use a more similar strategy to what a quantum computer would do,
# combine a set of offset diagonal matrices, each of which are constructed by testing the input index and applying
# a rotation to encode a cross section based on that index
def get_mass_matrix_LCU_vectors(D, L_f, mat_L, xs, BC="Dirichlet"):
    N = int(2**L_f) # size of the fine FEM cell discretization

    # N_f is the number of FEM basis functions per dimension (size of mass matrix per dimension)
    if BC == "Dirichlet":
        N_f = N - 1
    elif BC == "Vacuum":
        N_f = N + 1

    if BC == "Dirichlet":
        N_mat = int(2**mat_L) # size of the material grid in each dimension
        delta_N = int(2**(L_f - mat_L)) # number of dicrete cells in the FEM space per cell in the material grid
        offset_magnitudes = np.array([(N_f)**d for d in range(D-1,-1,-1)])
        xs_fine = np.kron(xs, np.ones((int(2**(L_f-mat_L)),) * D))
        mass_matrix_vecs = []

        ## OUTER LOOP: each iteration defines the values in a diagonal (potentially with an offset)
        for offsets in itertools.product([-1,0,1], repeat=D): # The 3^D different diagonals that make up the mass matrix
            idx_starts = [] # start indices for the submatrices to create the diagonals of the mass matrix
            nullified_indices = [] # indices where cross sections are "set to zero" because they are invalid values (outside domain etc)
            for d, offset in enumerate(offsets): # d=0 is most significant 
                if offset == -1:
                    idx_starts.append([0])
                    nullified_indices.append(0)
                if offset == 0:
                    idx_starts.append([0,1])
                    nullified_indices.append(-1) # dummy value becasue no indices are nullified
                if offset == 1:
                    idx_starts.append([1])
                    nullified_indices.append(N_f)
            scale_factor = 2**(sum([off == 0 for off in offsets])) / 6**D # multiplicative factor of 2 for non-offset diagonals (in each dimension), total matrix scaling factor of 1/6^D

            # INNER LOOP: Defines the terms within each diagonal (2^(number of 0 offsets)) per outer loop, 4^D total iterations
            for idx_start in itertools.product(*idx_starts):
                diag_vec = np.zeros((N_f)**D)
                for fine_index in range(len(diag_vec)): # iterate through each fine cell, very computationally inefficient classically, but quantumly this can be done coherently with all indices
                    fine_index_rolled = np.array(idx_start) + [(fine_index // (N_f)**d) % (N_f) for d in reversed(range(D))] # in a quantum circuit, the operations on fine_index are the operation on incoming bit string

                    # these rolled material indices can then be stored and controlled on to apply a rotation that encodes the cross section for that material region
                    mat_index_rolled = [fine_index_rolled[i] // delta_N for i in range(D)]

                    valid_index = True # qubit to determine whether to apply the xs
                    for d in range(D):
                        if(fine_index_rolled[d] == nullified_indices[d] and valid_index == True): # comparator gate between the fine index for dimension d and a preset register containing the integer nullified_indices[d]
                            valid_index = False # controlled on the result of the comparator gate, flip this boolean qubit

                    if valid_index:
                        diag_vec[fine_index] = xs[*mat_index_rolled] # xs rotation
                    else:
                        diag_vec[fine_index] = 0 # apply X Pauli gate to make the value 0 in the diagonal

                offset = sum(np.array(offsets) * offset_magnitudes)
                mass_matrix_vecs.append((scale_factor * diag_vec, offset))

        # combine the offset diagonal vectors into a matrix
        mass_matrix = np.zeros(((N_f)**D, (N_f)**D))
        for diag_vec,offset in mass_matrix_vecs:
            mass_matrix += np.roll(np.diag(diag_vec), shift=offset, axis=0)

    elif BC == "Vacuum":
        # TODO: figure out how to do the vacuum BC, now the FEM A matrix is bigger than the cell grid matrix, so instead of taking 2 shifted submatrices
        # for two different terms of the same diagonal when the offset is 0, you take the same entire matrix but offset within the diagonal
        N_mat = int(2**mat_L) # size of the material grid in each dimension
        delta_N = int(2**(L_f - mat_L)) # number of dicrete cells in the FEM space per cell in the material grid
        offset_magnitudes = np.array([(N_f)**d for d in range(D-1,-1,-1)])
        xs_fine = np.kron(xs, np.ones((int(2**(L_f-mat_L)),) * D))
        mass_matrix_vecs = []

        ## OUTER LOOP: each iteration defines the values in a diagonal (potentially with an offset)
        for offsets in itertools.product([-1,0,1], repeat=D): # The 3^D different diagonals that make up the mass matrix
            idx_starts = [] # start indices for the submatrices to create the diagonals of the mass matrix
            nullified_indices = [] # indices where cross sections are "set to zero" because they are invalid values (outside domain etc)
            for d, offset in enumerate(offsets): # d=0 is most significant 
                if offset == -1:
                    idx_starts.append([0])
                    nullified_indices.append(0)
                if offset == 0:
                    idx_starts.append([0,1])
                    nullified_indices.append(-1) # dummy value becasue no indices are nullified
                if offset == 1:
                    idx_starts.append([1])
                    nullified_indices.append(N_f)
            scale_factor = 2**(sum([off == 0 for off in offsets])) / 6**D # multiplicative factor of 2 for non-offset diagonals (in each dimension), total matrix scaling factor of 1/6^D

            # INNER LOOP: Defines the terms within each diagonal (2^(number of 0 offsets)) per outer loop, 4^D total iterations
            for idx_start in itertools.product(*idx_starts):
                diag_vec = np.zeros((N_f)**D)
                for fine_index in range(len(diag_vec)): # iterate through each fine cell, very computationally inefficient classically, but quantumly this can be done coherently with all indices
                    fine_index_rolled = np.array(idx_start) + [(fine_index // (N_f)**d) % (N_f) for d in reversed(range(D))] # in a quantum circuit, the operations on fine_index are the operation on incoming bit string

                    # these rolled material indices can then be stored and controlled on to apply a rotation that encodes the cross section for that material region
                    mat_index_rolled = [fine_index_rolled[i] // delta_N for i in range(D)]

                    valid_index = True # qubit to determine whether to apply the xs
                    for d in range(D):
                        if(fine_index_rolled[d] == nullified_indices[d] and valid_index == True): # comparator gate between the fine index for dimension d and a preset register containing the integer nullified_indices[d]
                            valid_index = False # controlled on the result of the comparator gate, flip this boolean qubit

                    if valid_index:
                        diag_vec[fine_index] = xs[*mat_index_rolled] # xs rotation
                    else:
                        diag_vec[fine_index] = 0 # apply X Pauli gate to make the value 0 in the diagonal

                offset = sum(np.array(offsets) * offset_magnitudes)
                mass_matrix_vecs.append((scale_factor * diag_vec, offset))

        # combine the offset diagonal vectors into a matrix
        mass_matrix = np.zeros(((N_f)**D, (N_f)**D))
        for diag_vec,offset in mass_matrix_vecs:
            mass_matrix += np.roll(np.diag(diag_vec), shift=offset, axis=0)
    return mass_matrix



D = 2 # dimensions
L = 3

# cross sections
# data I have been using: R=10. D1=0.1, D2=200, sigma_a_1=0.8, sigma_a_2=0.5, sigma_f_1=0.9, sigma_f_2=0.1
R = 1 # range of problem (in every dimension)
D_1 = 1.0 # diffusion coefficient in region 1 (1/cm)
D_2 = 10.0 # diffusion coefficient in region 2 (1/cm)
sigma_a_1 = 1.0 # macroscopic absorption cross section in region 1 (1/cm)
sigma_a_2 = 2.0 # macroscopic absorption cross section in region 2 (1/cm)
sigma_f_1 = 0.0 # macroscopic nu*fission cross section in region 1 (1/cm)
sigma_f_2 = 2.0 # macroscopic nu*fission cross section in region 2 (1/cm)
#diffusion_mat_base = np.array([D_1, D_2]) # 1D, mat_L=1
#absorption_xs_base = np.array([sigma_a_1, sigma_a_2]) # 1D, mat_L=1
#nu_fission_xs_base = np.array([sigma_f_1, sigma_f_2]) # 1D, mat_L=1
diffusion_mat_base = np.array([[D_1, D_2],[D_2, D_1]]) # 2D, mat_L=1
absorption_xs_base = np.array([[sigma_a_1, sigma_a_2],[sigma_a_2, sigma_a_1]]) # 2D, mat_L=1
nu_fission_xs_base = np.array([[sigma_f_1, sigma_f_2],[sigma_f_2, sigma_f_1]]) # 2D, mat_L=1
#diffusion_mat_base = np.array([[[D_1, D_2],[D_2, D_1]],[[D_2, D_1],[D_1, D_2]]]) # 3D, mat_L=1
#absorption_xs_base = np.array([[[sigma_a_1, sigma_a_2],[sigma_a_2, sigma_a_1]],[[sigma_a_2, sigma_a_1],[sigma_a_1, sigma_a_2]]]) # 3D, mat_L=1
#nu_fission_xs_base = np.array([[[sigma_f_1, sigma_f_2],[sigma_f_2, sigma_f_1]],[[sigma_f_2, sigma_f_1],[sigma_f_1, sigma_f_2]]]) # 3D, mat_L=1

# set up the 2D matrices of material properties (diffusion coefs and cross sections)
mat_L = 2 # the number of checkerboard spaces is 2^(D*mat_L)
'''if mat_L == 0:
    diffusion_mat = np.array([[D_1]])
    absorption_xs = np.array([[sigma_a_1]])
    nu_fission_xs = np.array([[sigma_f_1]])
else:
    diffusion_mat = np.kron(np.ones((2**(mat_L-1))), diffusion_mat_base)
    absorption_xs = np.kron(np.ones((2**(mat_L-1))), absorption_xs_base)
    nu_fission_xs = np.kron(np.ones((2**(mat_L-1))), nu_fission_xs_base)'''
if mat_L == 0:
    diffusion_mat = np.array([[D_1]])
    absorption_xs = np.array([[sigma_a_1]])
    nu_fission_xs = np.array([[sigma_f_1]])
else:
    diffusion_mat = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), diffusion_mat_base) # 2D expansion of material grid
    absorption_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), absorption_xs_base)
    nu_fission_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), nu_fission_xs_base)
'''if mat_L == 0:
    diffusion_mat = np.array([[D_1]])
    absorption_xs = np.array([[sigma_a_1]])
    nu_fission_xs = np.array([[sigma_f_1]])
else:
    diffusion_mat = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1),2**(mat_L-1))), diffusion_mat_base) # 3D expansion of material grid
    absorption_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1),2**(mat_L-1))), absorption_xs_base)
    nu_fission_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1),2**(mat_L-1))), nu_fission_xs_base)'''

N_1D = int(2**(L))
h = R/N_1D

# Dirichlet BCs
A_mat_old = get_2D_mass_matrix_LCU(D, L, np.kron(absorption_xs, np.ones((int(2**(L-mat_L)),) * D)))
A_mat = get_mass_matrix_LCU(D, L, mat_L, absorption_xs)
A_mat_diag_vecs_test = get_mass_matrix_LCU_vectors(D, L, mat_L, absorption_xs, BC='Dirichlet')
A_mat_diff = A_mat_diag_vecs_test - A_mat_old.toarray()

# Vacuum BCs
A_mat_diag_vecs_test = get_mass_matrix_LCU_vectors(D, L, mat_L, absorption_xs, BC="Vacuum")

