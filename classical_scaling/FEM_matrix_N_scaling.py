import sys
import numpy as np
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import math
import itertools
from fast_inversion.BPX import FEM_BPX_helpers as FEM


def main():
    D = 2
    L = 6
    N_1D = int(2**(L))
    N_total = int(N_1D**D)

    mat_L = 1
    diffusion_matrix = 2.5 * np.eye(int(2**(D*mat_L)))
    absorption_xs = np.array([[1.3, 1.2],[1.2, 1.3]]) # constant absorption xs
    absorption_xs = np.kron(absorption_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    nu_fission_xs = np.array([[0.9, 0.8],[0.8, 0.9]]) # constant fission xs
    nu_fission_xs = np.kron(nu_fission_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    L_mat = FEM.get_diffusion_matrix(D, L, diffusion_matrix)
    A_mat = FEM.get_mass_matrix_brute_force(L, D, absorption_xs)
    C_mat = FEM.get_mass_matrix_brute_force(L, D, nu_fission_xs)

    eigvals, eigvecs = eigh(L_mat+A_mat, C_mat, subset_by_index=[0,0])
    print(eigvals[0])
    
    main_eigvec = eigvecs
    main_eigvec = main_eigvec.reshape((N_1D-1,) * D)
    plt.imshow(main_eigvec, cmap='hot', interpolation='nearest') # 'cmap' specifies the color map

    plt.colorbar()
    plt.title('Neutron Scalar Flux')
    plt.xlabel('x index')
    plt.ylabel('y index')
    plt.show()
    
if __name__ == "__main__":
    main()
