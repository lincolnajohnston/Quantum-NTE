import sys
import numpy as np
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp2d
import scipy as sp
import math
import itertools
from fast_inversion.BPX import FEM_BPX_helpers as FEM
import time

def get_eigenvalue(D, L, mat_L, diffusion_mat, absorption_xs, nu_fission_xs, f, doPlot=False):
    N_1D = int(2**(L))
    h = 1/N_1D
    N_total = int(N_1D**D)
    start_time = time.perf_counter()

    diffusion_mat = np.kron(diffusion_mat, np.ones((int(2**(L-mat_L)),) * D))
    diffusion_matrix = np.diag(diffusion_mat.flatten())
    #diffusion_matrix = 2.5 * np.eye(int(2**(D*mat_L)))
    absorption_xs = np.kron(absorption_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    nu_fission_xs = np.kron(nu_fission_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    #L_mat_BPX = FEM.get_diffusion_matrix(D, L, diffusion_matrix)
    #P2 = np.kron(FEM.get_M1(L,np.ones(int(2**L))),FEM.get_P1(L,np.ones(int(2**L)))) + np.kron(FEM.get_P1(L,np.ones(int(2**L))),FEM.get_M1(L,np.ones(int(2**L))))
    #L_mat = FEM.get_2D_diffusion_matrix_brute_force(L, D, diffusion_mat)
    L_mat = FEM.get_2D_diffusion_matrix_LCU(D, L, diffusion_mat)
    diffusion_mat_time = time.perf_counter()

    # Assuming P2 is the correct diffusion matrix (but can only handle constant diffusion cross sections), what is the error of L_mat and L_mat_BPX
    '''L_mat_error = L_mat - P2
    L_mat_BPX_error = L_mat_BPX - P2
    L_mat_L2_error = np.linalg.norm(L_mat_error)
    L_mat_BPX_L2_error = np.linalg.norm(L_mat_BPX_error)'''

    #L_mat = L_mat_BPX
    #A_mat = FEM.get_mass_matrix_brute_force(L, D, absorption_xs)
    A_mat = FEM.get_2D_mass_matrix_LCU(D, L, absorption_xs)
    #C_mat = FEM.get_mass_matrix_brute_force(L, D, nu_fission_xs)
    C_mat = FEM.get_2D_mass_matrix_LCU(D, L, nu_fission_xs)
    mass_mat_time = time.perf_counter()

    K = (L_mat + A_mat).tocsr()
    M = C_mat.tocsr()

    # if a function was made for interpolating a coarser solution to get a good initial guess, use it
    # Finer grid
    finer_x = np.linspace(0, 1, N_1D-1)
    finer_y = np.linspace(0, 1, N_1D-1)
    # Interpolate
    finer_mesh = f(finer_x, finer_y)
    eigvec_guess = finer_mesh.flatten()

    if doPlot:
        eigvals, eigvecs = eigsh(K,k=1,M=M,sigma=0.0,which='LM',tol=1e-6, v0 = eigvec_guess) # smallest eigenvalue (reactor fundamental mode)
    else:
        eigvals = eigsh(K,k=1,M=M,sigma=0.0,which='LM',tol=1e-6, return_eigenvectors=False) # smallest eigenvalue (reactor fundamental mode)
    eig_solve_time = time.perf_counter()

    #eigvals_slow, eigvecs_slow = eigh(L_mat.toarray() + A_mat.toarray(), C_mat.toarray(), eigvals_only=False)
    if doPlot:
        main_eigvec = eigvecs
        main_eigvec = main_eigvec.reshape((N_1D-1,) * D)

        # get a good initial guess for the next level by doing interpolation of the current level (only set up for 2D here)
        # Original coordinates
        x = np.linspace(0, 1, N_1D-1)
        y = np.linspace(0, 1, N_1D-1)
        z = main_eigvec
        f = interp2d(x, y, z, kind='linear')

        plt.imshow(main_eigvec, cmap='jet', interpolation='nearest')

        plt.colorbar()
        plt.title('Neutron Scalar Flux')
        plt.xlabel('x index')
        plt.ylabel('y index')
        plt.show()

    plot_time = time.perf_counter()
    print("Diffusion matrix time: ", diffusion_mat_time - start_time)
    print("Mass matrix time: ", mass_mat_time - diffusion_mat_time)
    print("Eigenvalue solve time: ", eig_solve_time - diffusion_mat_time)
    print("Plot time: ", plot_time - eig_solve_time)

    return eigvals[0], f

def main():
    D = 2
    L_list = range(3,9)
    mat_L = 1
    diffusion_mat = np.array([[1, 200],[200, 1]]) # 2D
    #diffusion_mat = np.array([[0.1, 200, 0.1, 200],[200, 0.1, 200, 0.1],[0.1, 200, 0.1, 200],[200, 0.1, 200, 0.1],]) # 2D
    #diffusion_mat = np.array([1,2]) # 1D
    absorption_xs = np.array([[0.7, 0.6],[0.6, 0.7]])
    #absorption_xs = np.array([[0.8, 0.8, 0.8, 0.8],[0.8, 0.8, 0.8, 0.8],[0.8, 0.8, 0.8, 0.8],[0.8, 0.8, 0.8, 0.8]])
    #absorption_xs = np.array([[0, 0],[0, 0]])
    nu_fission_xs = np.array([[0.9, 0.8],[0.8, 0.9]])
    #nu_fission_xs = np.array([[0.9, 0.9,0.9,0.9],[0.9, 0.9,0.9,0.9],[0.9, 0.9,0.9,0.9],[0.9, 0.9,0.9,0.9]])

    # set up initial guess
    x = np.linspace(0, 1, 2)
    y = np.linspace(0, 1, 2)
    z = np.ones(4)
    f = interp2d(x, y, z, kind='linear')


    eigenvalue_list = []
    for L in L_list:
        eigenvalue, f = get_eigenvalue(D, L, mat_L, diffusion_mat, absorption_xs, nu_fission_xs, f, doPlot=True)
        eigenvalue_list.append(eigenvalue)
    p = []
    for i in range(2,len(eigenvalue_list)):
        p.append(math.log((eigenvalue_list[i-2] - eigenvalue_list[i-1]) / (eigenvalue_list[i-1] - eigenvalue_list[i])) / math.log(4))
    plt.plot(L_list, eigenvalue_list)
    plt.title("lambda eigenvalue vs L")
    plt.xlabel("L")
    plt.ylabel("lambda eigenvalue (1/k-eigenvalue)")
    plt.show()

    plt.plot(L_list[2:], p)
    plt.title(r"p value in $\epsilon=h^{1/p}$ vs L")
    plt.xlabel("L")
    plt.ylabel("p")
    plt.show()

    print("L values: ", list(L_list[2:]))
    print("p values: ", p)
    
    
if __name__ == "__main__":
    main()
