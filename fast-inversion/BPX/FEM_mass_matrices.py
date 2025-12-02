import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools
import FEM_BPX_helpers as FEM

# Using the FEM discretization of the diffusion equation, we get scaled mass matrices for both the fission term and the absorption term
# This script just writes out these matrices to get insight into how we are going to quantumly encode them

# assume we have the same domain size and level of discretization in each spatial dimension

class MassMatrix:
    def __init__(self, input_data):
        self.N = np.array(input_data["n"])
        self.D = len(self.N)
        self.h = np.array(input_data["h"])

    # return a mass matrix where each term is weighted by a piecewise-constant value (like the absorption cross section)
    # D is number of dimensions, L is number of levels, consts is an array of constants defined in each cell of the FEM discretization
    def get_1D_weighted_mass_matrix(self, L, consts):
        # make sure consts is the right size
        if len(consts) != int(2**(L)):
            raise ValueError("consts matrix is not the correct size/shape")
        A = np.diag(2*consts[:-1])
        A += np.diag(2*consts[1:])
        A += np.diag(consts[1:-1], k=1)
        A += np.diag(consts[1:-1], k=-1)
        return A


    # convert multidimensional (x1,x2...,xD) index to 1D index, first index is most significant
    def unroll_index(self, index_vec, xs_mesh=False):
        roll_N = self.N + 1 if xs_mesh else self.N
        return sum([index_vec[d]*math.prod(roll_N[d+1:]) for d in range(len(index_vec))])

    # convert 1D index to multidimensional (x1,x2...,xD) index, first index is most significant
    def roll_index(self, index, xs_mesh=False):
        indices = np.zeros(self.D, dtype=int) # self.dim spatial dimensions
        roll_N = self.N + 1 if xs_mesh else self.N
        for d in range(self.D):
            indices[d] = index
            if d < self.D:
                indices[d] = math.floor(indices[d] / math.prod(roll_N[d+1:]))
            if d > 0:
                indices[d] = indices[d] % roll_N[d-1]
        return indices

    # assume domain is 1 so h = 1/2^L, factor out the h^D term
    def get_mass_matrix_brute_force(self, L, xs):
        M = np.zeros((np.prod(self.N), np.prod(self.N)))

        match_term = 1/3 # integral over a cell when the hat functions are matching
        offset_term = 1/6 # integral over a cell when one hat function is increasing and one is decreasing

        all_indices = itertools.product(list(range(N)), repeat=self.D)
        for node_index in all_indices: # iterate through all nodes
            offset_indices = itertools.product(list(range(-1,2)), repeat=self.D)
            for node_offset_index in offset_indices: # iterate through each node surrounding the current node
                    row_index = np.array(node_index)
                    col_index = row_index + np.array(node_offset_index)

                    # skip the nodes that are outside the domain
                    valid_index = True
                    for d in range(self.D):
                        if col_index[d] < 0 or col_index[d] >= N:
                            valid_index = False
                    if not valid_index:
                        continue

                    xs_coef = 2**(D-sum(np.abs(np.array(node_offset_index)))) / 6**self.D # coefficient on each of the cross sections in the matrix
                    sigma_index_lower = row_index + [max(offset, 0) for offset in node_offset_index] # lower index of the xs terms
                    sigma_index_upper = [min(col_index[i], row_index[i]) + 1 for i in range(len(row_index))] # upper index of the xs terms
                    xs_indices_list = [list(range(sigma_index_lower[d], sigma_index_upper[d]+1)) for d in range(self.D)]
                    xs_indices = itertools.product(*xs_indices_list)
                    for xs_index in xs_indices:
                        M[self.unroll_index(row_index), self.unroll_index(col_index)] += xs_coef * xs[xs_index]
        return M
    
    # given a cross section vector, xs, and a list of index ranges ([[x_low, x_high], [y_low, y_high], ...])
    # return the vector of cross sections only within those index ranges
    def get_xs_subvector(self, xs_list, index_ranges):
        return np.array(xs_list[np.ix_(*[list(range(index_ranges[d][0], index_ranges[d][1]+1)) for d in range(self.D)])]).flatten()
    
    # assume domain is 1 so h = 1/2^L, factor out the h^D term
    # craft the matrix by taking the linear combination of diagonal matrices, each of which 
    # can be efficiently block-encoded and then ocmbined with LCU.
    # This function is uncompleted, just here to show that the fission and absorption FEM
    # matrices can be implemented as the linear combination of 6^D diagonal matrices (with some integer shift |x> -> |x+1>)
    def get_mass_matrix_LCU(self, D, L, xs):
        M = np.zeros((np.prod(self.N), np.prod(self.N)))

        # D overlapping hat functions
        # main diagonal, for D>=0
        coef = (1/3)**D # each overlapping hat function's integral product is 1/3
        all_indices = itertools.product(list(range(2)), repeat=self.D) # starting indices for xs list
        for start_indices in all_indices:
            diag = self.get_xs_subvector(xs, [[start_indices[d], self.N[d]-1+start_indices[d]] for d in range(D)]) # get cross sections for all but one index in each dimension (0...N-1 or 1...N)
            M += coef * np.diag(diag) # add the cross secitons to the main diagonal
            #print(diag)

        # 1 offset hat function, D-1 overlapping hat functions
        coef = (1/3)**(D-1) * (1/6)**1 # offset hat functions have a square integral of 1/6
        # LSB (least significant bit) increment, for D > 0
        all_indices = itertools.product(list(range(2)), repeat=self.D-1) # just the index range for the more significant bits
        for start_indices in all_indices:
            diag = self.get_xs_subvector(xs, [[start_indices[d], self.N[d]-1+start_indices[d]] for d in range(D-1)] + [[1,self.N[D-1]]])
            M += coef * np.diag(diag)
            #print(diag)
        
        return M
    
            

L_vals = list(range(2,5))
mat_L = 2
D = 2

for L in L_vals:
    print("Level: ", L)
    h = 1/2**L
    N = int(1/h - 1)

    mass_mat = MassMatrix(input_data={"n":[N]*D, "h":[h]*D})

    absorption_vec_small_1D = np.random.rand(2**(mat_L)) # random absorption cross sections
    #absorption_vec_small = np.ones(int(2**(D*mat_L))) # all ones absorption cross sections

    absorption_vec_small = np.array(range(1,2**(mat_L*D)+1)).reshape([int(2**mat_L)]*D)
    #absorption_vec_small = np.random.rand(*([2**mat_L]*D)) # random absorption cross sections
    absorption_vec_large = np.kron(absorption_vec_small, np.ones([int(2**(L-mat_L))]*D))

    #absorption_vec_large = np.array(range(2**(L*D)))
    B = h**D * mass_mat.get_mass_matrix_brute_force(L, absorption_vec_large)
    B_inv = np.linalg.inv(B)
    B_LCU = h**D * mass_mat.get_mass_matrix_LCU(D, L, absorption_vec_large)

    diffusion_mat_small = np.eye(int(2**(D*mat_L))) # all ones diffusion coefficients
    diffusion_mat = np.kron(diffusion_mat_small, np.eye(2**(D*(L - mat_L))))
    D_A = np.kron(diffusion_mat, np.eye(D))
    C_L = FEM.getC_l(D, L)
    A = np.transpose(C_L) @ np.kron(D_A, np.eye(2**D)) @ C_L # diffusion matrix
    A_inv = np.linalg.inv(A)
    C = (A+B)
    C_inv = np.linalg.inv(C)
    G = np.eye(len(B)) + A_inv @ B
    G_inv = np.linalg.inv(G)

    B_sing_max = np.linalg.norm(B, ord=2)
    B_sing_min = 1/np.linalg.norm(B_inv, ord=2)
    A_sing_max = np.linalg.norm(A, ord=2)
    A_sing_min = 1/np.linalg.norm(A_inv, ord=2)
    C_sing_max = np.linalg.norm(C, ord=2)
    C_sing_min = 1/np.linalg.norm(C_inv, ord=2)
    G_sing_max = np.linalg.norm(G, ord=2)
    G_sing_min = 1/np.linalg.norm(G_inv, ord=2)
    G_sing_min_lower_bound_A = 1/(1 + (1/C_sing_min) * A_sing_max)
    G_sing_min_lower_bound_B = 1/(1 + (1/C_sing_min) * B_sing_max)

    print("All Singular Values: ")
    print("M max singular value: ", B_sing_max)
    print("M min singular value: ", B_sing_min)
    print("L max singular value: ", A_sing_max)
    print("L min singular value: ", A_sing_min)
    print("C max singular value: ", C_sing_max)
    print("C min singular value: ", C_sing_min)
    print("G max singular value: ", G_sing_max)
    print("G min singular value: ", G_sing_min)
    print("")
    print("Values relevant to the fast-inverse theorem")
    print("Norm of A inverse (lower bound of alpha'_A): ", 1/A_sing_min)
    print("Norm of B (lower bound of alpha_B): ", B_sing_max)
    print("Minimum singular value of (I + A^-1 B), sigma_min: ", G_sing_min)
    print("Proposed lower bound for sigma_min using A norm: ", G_sing_min_lower_bound_A)
    print("Proposed lower bound for sigma_min using B norm: ", G_sing_min_lower_bound_B)
    print("------------------------------------------------------------\n")

    #B = mass_mat.get_1D_weighted_mass_matrix(L, np.kron(absorption_vec_small_1D, np.ones(2**((L - mat_L))))) # absorption cross section FEM operator
    #print(B)