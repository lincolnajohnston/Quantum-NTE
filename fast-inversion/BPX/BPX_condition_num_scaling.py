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

# return a mass matrix where each term is weighted by a piecewise-constant value (like the absorption cross section)
# D is number of dimensions, L is number of levels, consts is an array of constants defined in each cell of the FEM discretization
def get_weighted_mass_matrix(D, L, consts):
    # make sure consts is the right size
    if consts.size() != tuple([int(2**L)]*D):
        raise ValueError("consts matrix is not the correct size/shape")
    A = np.diag(2*consts[:-1])
    A += np.diag(2*consts[1:])
    A += np.diag(consts[1:-1], k=1)
    A += np.diag(consts[1:-1], k=-1)
    return A

# the domain goes from 0 to 1
D_min = 2
D_max = 2
L_min = 2
L_max = 8
D_vals = np.array(range(D_min,D_max + 1))
L_vals = np.array(range(L_min,L_max + 1)) # number of levels of BPX preconditioner
D_and_L = [D_vals, L_vals]
FSF_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_conds = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
FSF_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
FSF_inv_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
S_inv_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))
C_F_inv_norms = np.zeros((D_max - D_min + 1, L_max - L_min + 1))

mat_L = 2
#diffusion_mat_small = [np.diag(np.random.rand(2**(D*mat_L))) for D in D_vals]
diffusion_mat_small = [np.eye(int(2**(D*mat_L))) for D in D_vals] # all ones diffusion coefficients

absorption_vec_small = [np.random.rand(2**(D*mat_L)) for D in D_vals] # random absorption cross sections
#absorption_vec_small = [np.ones(int(2**(D*mat_L))) for D in D_vals] # all ones absorption cross sections

for combo in itertools.product(*D_and_L):
    D = combo[0]
    L = combo[1]
    # find C_L (C_l for the finest level)
    C_L = getC_l(D, L)
    F = get_F(D, L)
    C_F = C_L @ F

    F_prime = get_F_prime(D,L)
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

    # testing norm of each section of F and how that compares to the norm of the entire F
    '''CF_col_sections = [(2**l-1)**D for l in range(1,L+1)]
    F_norm = np.linalg.norm(F.toarray(), ord=2)
    print("F norm: ", F_norm)
    for l in range(L):
        F_sub = F[:,sum(CF_col_sections[0:l]):sum(CF_col_sections[0:l+1])]
        F_sub_norm = np.linalg.norm(F_sub.toarray(), ord=2)
        print("Level ", l, " F sub norm: ", F_sub_norm)'''

    diffusion_mat = np.kron(diffusion_mat_small[D-D_min], np.eye(2**(D*(L - mat_L))))
    #diffusion_mat = np.diag(np.concatenate((3*np.ones(2**(D*L-1)),5*np.ones(2**(D*L-1))))) # matrix of diffusion coefficients
    D_A = np.kron(diffusion_mat, np.eye(D))

    S = np.transpose(C_L) @ np.kron(D_A, np.eye(2**D)) @ C_L
    FSF = np.transpose(F) @ S @ F # preconditioned system using the F matrix

    B = get_weighted_mass_matrix(D, np.kron(absorption_vec_small[D-D_min], np.ones(2**(D*(L - mat_L))))) # absorption cross section FEM operator
    #M = get_weighted_mass_matrix(D, L-mat_L, np.kron(np.ones(int(2**(D*mat_L))), np.ones(2**(D*(L - mat_L))))) # mass matrix

    # check if S and FSF are normal matrices
    T1 = np.transpose(S) @ S
    T2 = S @ np.transpose(S)
    print("S is " + ("" if np.linalg.norm(T1-T2) < 1E-10 else "not ") + "normal for D=" + str(D) + " and L="+str(L))

    T1 = np.transpose(FSF) @ FSF
    T2 = FSF @ np.transpose(FSF)
    print("FSF is " + ("" if np.linalg.norm(T1-T2) < 1E-10 else "not ") + "normal for D=" + str(D) + " and L="+str(L))

    _, FSF_sing_vals, _ = np.linalg.svd(FSF)
    _, S_sing_vals, _ = np.linalg.svd(S)
    _, C_F_sing_vals, _ = np.linalg.svd(C_F.toarray())
    S_sing_vals = S_sing_vals[abs(S_sing_vals) > 1E-12]
    FSF_sing_vals = FSF_sing_vals[abs(FSF_sing_vals) > 1E-12]
    C_F_sing_vals = C_F_sing_vals[abs(C_F_sing_vals) > 1E-12]


    FSF_norm = np.max(FSF_sing_vals)
    FSF_inv_norm = np.min(FSF_sing_vals)
    FSF_cond = FSF_norm / FSF_inv_norm
    #print("FSF norm: ", FSF_norm)
    #print("FSF_inv norm: ", FSF_inv_norm)
    #print("FSF cond: ", FSF_cond)

    S_norm = np.max(S_sing_vals)
    S_inv_norm = np.min(S_sing_vals)
    S_cond = S_norm / S_inv_norm
    #print("\nS norm: ", S_norm)
    #print("S_inv norm: ", S_inv_norm)
    #print("S cond: ", S_cond)

    C_F_norm = np.max(C_F_sing_vals)
    C_F_inv_norm = np.min(C_F_sing_vals)
    C_F_cond = C_F_norm / C_F_inv_norm
    #print("\nS norm: ", S_norm)
    #print("S_inv norm: ", S_inv_norm)
    #print("S cond: ", S_cond)

    # store the condition numbers
    FSF_conds[D-D_min,L-L_min] = FSF_cond
    S_conds[D-D_min,L-L_min] = S_cond
    C_F_conds[D-D_min,L-L_min] = C_F_cond

    # store the matrix norms
    FSF_norms[D-D_min,L-L_min] = FSF_norm
    S_norms[D-D_min,L-L_min] = S_norm
    C_F_norms[D-D_min,L-L_min] = C_F_norm

    # store the matrix norms of the inverse of the matrice
    FSF_inv_norms[D-D_min,L-L_min] = FSF_inv_norm
    S_inv_norms[D-D_min,L-L_min] = S_inv_norm
    C_F_inv_norms[D-D_min,L-L_min] = C_F_inv_norm

for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, FSF_conds[D-D_min,:])
    plt.title("Condition Numbers vs L")
    plt.xlabel("L")
    print("FSF d = " + str(D) + " norms: ", FSF_norms[D-D_min,:])
    print("FSF d = " + str(D) + " inverse norms: ", FSF_inv_norms[D-D_min,:])
    print("FSF d = " + str(D) + " condition numbers: ", FSF_conds[D-D_min,:])

for D in range(D_min, D_max + 1):
    plt.semilogy(L_vals, S_conds[D-D_min,:])
    print("S d = " + str(D) + " norms: ", S_norms[D-D_min,:])
    print("S d = " + str(D) + " inverse norms: ", S_inv_norms[D-D_min,:])
    print("S d = " + str(D) + " condition numbers: ", S_conds[D-D_min,:])

for D in range(D_min, D_max + 1):
    print("C_F d = " + str(D) + " norms: ", C_F_norms[D-D_min,:])
    print("C_F d = " + str(D) + " inverse norms: ", C_F_inv_norms[D-D_min,:])
    print("C_F d = " + str(D) + " condition numbers: ", C_F_conds[D-D_min,:])

plt.legend(["FSF matrix D=" + str(d) for d in range(D_min, D_max + 1)] + ["S matrix D=" + str(d) for d in range(D_min, D_max + 1)])
plt.show()