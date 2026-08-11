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
import time

def get_eigenvalue(D, L, mat_L, R, diffusion_mat, absorption_xs, nu_fission_xs, sigma=0.0, doPlot=False):
    N_1D = int(2**(L))
    h = R/N_1D
    N_total = int(N_1D**D)
    start_time = time.perf_counter()

    # expand the material data to the finer grid (level L from level mat_L)
    diffusion_mat = np.kron(diffusion_mat, np.ones((int(2**(L-mat_L)),) * D))
    absorption_xs = np.kron(absorption_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))
    nu_fission_xs = np.kron(nu_fission_xs, np.ones((int(2**(L-mat_L)), int(2**(L-mat_L)))))

    ##########find the diffusion matrix ##########
    #L_mat_BPX = FEM.get_diffusion_matrix(D, L, diffusion_matrix)
    #P2 = np.kron(FEM.get_M1(L,np.ones(int(2**L))),FEM.get_P1(L,np.ones(int(2**L)))) + np.kron(FEM.get_P1(L,np.ones(int(2**L))),FEM.get_M1(L,np.ones(int(2**L))))
    #L_mat_slow = FEM.get_2D_diffusion_matrix_brute_force(L, D, diffusion_mat)
    L_mat = FEM.get_2D_diffusion_matrix_LCU(D, L, diffusion_mat)
    #L_mat_diff = L_mat_slow - L_mat
    #L_mat_L2_error = np.linalg.norm(L_mat_diff.toarray())
    diffusion_mat_time = time.perf_counter()

    ##########find the absorption and fission matrices ##########
    #A_mat_slow = h**2 * FEM.get_mass_matrix_brute_force(L, D, absorption_xs)
    A_mat = h**2 * FEM.get_2D_mass_matrix_LCU(D, L, absorption_xs)
    #A_mat_diff = A_mat_slow - A_mat
    #A_mat_L2_error = np.linalg.norm(A_mat_diff.toarray())
    #C_mat_slow = h**2 * FEM.get_mass_matrix_brute_force(L, D, nu_fission_xs)
    C_mat = h**2 * FEM.get_2D_mass_matrix_LCU(D, L, nu_fission_xs)
    #C_mat_diff = C_mat_slow - C_mat
    #C_mat_L2_error = np.linalg.norm(C_mat_diff.toarray())
    mass_mat_time = time.perf_counter()

    K = (L_mat + A_mat).tocsr() # left side matrix of generalized eigenvalue problem
    M = C_mat.tocsr() # right side matrix of generalized eigenvalue problem

    if doPlot:
        eigvals, eigvecs = eigsh(K,k=1,M=M,sigma=sigma,which='LM') # find smallest lambda eigenvalue (reactor fundamental mode)
    else:
        eigvals = eigsh(K,k=1,M=M,sigma=sigma,which='LM', return_eigenvectors=False) # find smallest lambda eigenvalue (reactor fundamental mode)
    eig_solve_time = time.perf_counter()

    #eigvals_slow, eigvecs_slow = eigh(L_mat.toarray() + A_mat.toarray(), C_mat.toarray(), eigvals_only=False) # calculate the eigenvalues the slow way

    # plot the eigenvector
    if doPlot:
        main_eigvec = eigvecs
        main_eigvec = main_eigvec.reshape((N_1D-1,) * D)

        plt.imshow(main_eigvec, cmap='jet', interpolation='nearest')

        plt.colorbar()
        plt.title('Neutron Scalar Flux')
        plt.xlabel('x index')
        plt.ylabel('y index')
        plt.show()

    # print time it took to perform each task
    plot_time = time.perf_counter()
    print("----------------------------------")
    print("L = ", L)
    print("Diffusion matrix time: ", diffusion_mat_time - start_time)
    print("Mass matrix time: ", mass_mat_time - diffusion_mat_time)
    print("Eigenvalue solve time: ", eig_solve_time - diffusion_mat_time)
    print("Plot time: ", plot_time - eig_solve_time)
    print("----------------------------------\n")

    return eigvals[0]

def main():
    D = 2 # dimensions
    L_list = range(4,11) # list of FEM levels to test

    # cross sections
    R = 10 # range of problem (in every dimension)
    D_1 = 0.1 # diffusion coefficient in region 1 (1/cm)
    D_2 = 200 # diffusion coefficient in region 2 (1/cm)
    sigma_a_1 = 0.8 # macroscopic absorption cross section in region 1 (1/cm)
    sigma_a_2 = 0.5 # macroscopic absorption cross section in region 2 (1/cm)
    sigma_f_1 = 0.9 # macroscopic nu*fission cross section in region 1 (1/cm)
    sigma_f_2 = 0.1 # macroscopic nu*fission cross section in region 2 (1/cm)
    diffusion_mat_base = np.array([[D_1, D_2],[D_2, D_1]]) # 2D, mat_L=1
    absorption_xs_base = np.array([[sigma_a_1, sigma_a_2],[sigma_a_2, sigma_a_1]]) # 2D, mat_L=1
    nu_fission_xs_base = np.array([[sigma_f_1, sigma_f_2],[sigma_f_2, sigma_f_1]]) # 2D, mat_L=1

    # set up the 2D matrices of material properties (diffusion coefs and cross sections)
    mat_L = 2 # the number of checkerboard spaces is 2^(D*mat_L)
    diffusion_mat = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), diffusion_mat_base)
    absorption_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), absorption_xs_base)
    nu_fission_xs = np.kron(np.ones((2**(mat_L-1),2**(mat_L-1))), nu_fission_xs_base)

    # calculate the eigenvalues for each of the Levels being tested
    eigenvalue_list = []
    for L in L_list:
        # calculate sigma, a lower bound on the lambda eigenvalue, using the previous eigenvalues found
        if len(eigenvalue_list) > 1:
            sigma = eigenvalue_list[-1] - 5*(eigenvalue_list[-2] - eigenvalue_list[-1]) # if there are at least two previous eigenvalues calculated, use those to find a lower bound on the next eigenvalue (this is not precise)
        else:
            sigma=0
        eigenvalue = get_eigenvalue(D, L, mat_L, R, diffusion_mat, absorption_xs, nu_fission_xs, sigma=sigma, doPlot=True) # create the problem matrices and return the principal eigenvalue
        eigenvalue_list.append(eigenvalue)

    # find the exponents, p, on h using the eigenvalues found
    p = []
    for i in range(2,len(eigenvalue_list)):
        p.append(math.log((eigenvalue_list[i-2] - eigenvalue_list[i-1]) / (eigenvalue_list[i-1] - eigenvalue_list[i])) / math.log(4))


    # plot results
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


    # print results
    print("L values: ", list(L_list[2:]))
    print("p values: ", p, "\n")

    print("L values: ", list(L_list))
    print("lambda eigenvalues: ", eigenvalue_list)
    print("done")
    
    
if __name__ == "__main__":
    main()
