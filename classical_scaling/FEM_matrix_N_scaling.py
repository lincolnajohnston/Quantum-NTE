import sys
import numpy as np
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
import scipy as sp
import math
import itertools
from fast_inversion.BPX import FEM_BPX_helpers as FEM

def main():
    D = 2
    L = 6
    N_1D = int(2**(L))
    h = 1/N_1D
    N_total = int(N_1D**D)

    mat_L = 1
    diffusion_mat = np.array([[0.1, 0.2],[0.2, 0.1]]) # 2D
    #diffusion_mat = np.array([1,2]) # 1D
    diffusion_mat = np.kron(diffusion_mat, np.ones((int(2**(L-mat_L)),) * D))
    diffusion_matrix = np.diag(diffusion_mat.flatten())
    #diffusion_matrix = 2.5 * np.eye(int(2**(D*mat_L)))
    absorption_xs = np.array([[0.8, 0.8],[0.8, 0.8]])
    #absorption_xs = np.array([[0, 0],[0, 0]])
    absorption_xs = np.kron(absorption_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    nu_fission_xs = np.array([[0.9, 0.9],[0.9, 0.9]])
    nu_fission_xs = np.kron(nu_fission_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    L_mat_BPX = FEM.get_diffusion_matrix(D, L, diffusion_matrix)
    L_mat = FEM.get_2D_diffusion_matrix_brute_force(L, D, diffusion_mat)
    #L_mat = L_mat_BPX
    A_mat = FEM.get_mass_matrix_brute_force(L, D, absorption_xs)
    C_mat = FEM.get_mass_matrix_brute_force(L, D, nu_fission_xs)

    K = (L_mat + A_mat).tocsr()
    M = C_mat.tocsr()

    eigvals, eigvecs = eigsh(
        K,
        k=1,
        M=M,
        sigma=0.0,      # smallest eigenvalue (reactor fundamental mode)
        which='LM'
    )

    #eigvals_slow, eigvecs_slow = eigh(L_mat.toarray() + A_mat.toarray(), C_mat.toarray(), eigvals_only=False)
    print(eigvals[0])
    
    main_eigvec = eigvecs
    main_eigvec = main_eigvec.reshape((N_1D-1,) * D)
    plt.imshow(main_eigvec, cmap='jet', interpolation='nearest')

    plt.colorbar()
    plt.title('Neutron Scalar Flux')
    plt.xlabel('x index')
    plt.ylabel('y index')
    plt.show()
    
if __name__ == "__main__":
    main()
