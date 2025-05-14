import sys
import os
sys.path.append(os.getcwd())
import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

############# 1D Diffusion eigenvalue results ###############
# Dimensions: 1
# Equation Diffusion
# Geometry: Square Fuel Pin
# Notes: 
n_dim = 1
input_folder = 'simulations/ProblemData_1D_scaling_tests_fuel_pin/'
min_qubits = 1
max_qubits = 10
max_dim_size = int(math.pow(2,max_qubits))
ranges = [4.8] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'
invert_equation = True
fund_eig_index = -1 if invert_equation else 0

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
A_max_sings = np.zeros(max_qubits - min_qubits + 1)
A_min_sings = np.zeros(max_qubits - min_qubits + 1)
B_max_sings = np.zeros(max_qubits - min_qubits + 1)
B_min_sings = np.zeros(max_qubits - min_qubits + 1)
condAs = np.zeros(max_qubits - min_qubits + 1)
condBs = np.zeros(max_qubits - min_qubits + 1)
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    if invert_equation:
        B_matrix, A_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    else:
        A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)

    # print statements for testing
    if(False):
        if np.linalg.det(A_matrix) != 0:
            A_min_eig = np.min(np.linalg.eig(A_matrix)[0])
            A_max_eig = np.max(np.linalg.eig(A_matrix)[0])
            condA = np.linalg.cond(A_matrix)
            condAinv = np.linalg.cond(np.linalg.inv(A_matrix))
            condAs[i - min_qubits] = condA
        else:
            A_min_eig = 0
            A_max_eig = np.inf
            condA = np.inf
            condAinv = "DNE"
        if np.linalg.det(B_matrix) != 0:
            B_min_eig = np.min(np.linalg.eig(B_matrix)[0])
            B_max_eig = np.max(np.linalg.eig(B_matrix)[0])
            condB = np.linalg.cond(B_matrix)
            condBinv = np.linalg.cond(np.linalg.inv(B_matrix))
            condBs[i - min_qubits] = condB
        else:
            B_min_eig = 0
            B_max_eig = np.inf
            condB = np.inf
            condBinv = "DNE"
        #A_inv = np.linalg.inv(A_matrix)
        B_inv = np.linalg.inv(B_matrix)
        A_norm = np.max(svdvals(A_matrix))
        A_inv_norm = np.min(svdvals(A_matrix))
        B_norm = np.max(svdvals(B_matrix))
        B_inv_norm = np.min(svdvals(B_matrix))
        A_max_sings[i - min_qubits] = A_norm
        B_max_sings[i - min_qubits] = B_norm
        A_min_sings[i - min_qubits] = A_inv_norm
        B_min_sings[i - min_qubits] = B_inv_norm
        print("A min eig: ", A_min_eig)
        print("A max eig: ", A_max_eig)
        print("B min eig: ", B_min_eig)
        print("B max eig: ", B_max_eig)
        print("A condition number: ", condA)
        print("B condition number: ", condB)
        print("A_norm: ", A_norm)
        print("B_norm: ", B_norm)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[fund_eig_index]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,fund_eig_index].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,fund_eig_index] / abs(eigvecs[0,fund_eig_index]) # scale eigenvector and make first value positive
    eigenvector_results[i - min_qubits,:] = eigenvector_results[i - min_qubits,:] / np.linalg.norm(eigenvector_results[i - min_qubits,:]) # normalize eigenvectors

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])


############# 2D Diffusion eigenvalue results ###############
# Dimensions: 2
# Equation Diffusion
# Geometry: Square Fuel Pin
# Notes: 
'''n_dim = 2
input_folder = 'simulations/ProblemData_2D_scaling_tests_fuel_pin/'
min_qubits = 1
max_qubits = 5
max_dim_size = int(math.pow(2,max_qubits))
ranges = [4.0, 4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'
invert_equation = True
fund_eig_index = -1 if invert_equation else 0

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    if invert_equation:
        B_matrix, A_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    else:
        A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[fund_eig_index]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,fund_eig_index].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,fund_eig_index] / abs(eigvecs[0,fund_eig_index]) # scale eigenvector and make first value positive
    eigenvector_results[i - min_qubits,:] = eigenvector_results[i - min_qubits,:] / np.linalg.norm(eigenvector_results[i - min_qubits,:]) # normalize eigenvectors

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''



############# 3D Diffusion eigenvalue results ###############
# Dimensions: 3
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: still shows the h^2 dependence like in the 1D case and the same as the local truncation error
'''n_dim = 3
input_folder = 'simulations/ProblemData_3D_scaling_tests/'
min_qubits = 1
max_qubits = 4
max_dim_size = int(math.pow(2,max_qubits))
ranges = [4.0, 4.0, 4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,0].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # scale eigenvector and make first value positive

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''


############# 2D Diffusion eigenvalue results ###############
# Dimensions: 2
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: still shows the h^2 dependence like in the 1D case and the same as the local truncation error
'''n_dim = 2
input_folder = 'simulations/ProblemData_2D_scaling_tests/'
min_qubits = 1
max_qubits = 5
max_dim_size = int(math.pow(2,max_qubits))
ranges = [4.0, 4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,0].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # scale eigenvector and make first value positive

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''


############# 1D Diffusion eigenvalue results ###############
# Dimensions: 1
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: shows h^2 dependence of eigenvalue error, same as the local truncation error, h dependence for the L2 of the eigenvector
# Notes: Benchmark PUa-l-0-SL from https://www.sciencedirect.com/science/article/pii/S0149197002000987
# Notes: if you set the x_range to a large number, we get a k-eig almost exactly right at 2.613 (compared to the PUa-l-O-IN benchmark's 2.6129)
# Notes: but the finite slab benchmarks has significantly different eigenvalues
'''input_folder = 'simulations/ProblemData_1D_scaling_tests/'
n_dim = 1
min_qubits = 1
max_qubits = 8
max_dim_size = int(math.pow(2,max_qubits))
x_range = 3.7074
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = x_range / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,0].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # extend/interpolate eigenvectors onto finest grid and make first value positive

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''


############# 1D Diffusion FD eigenvalue results ###############
# Dimensions: 1
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Discretization Scheme: Finite Difference
# Notes: USES 0 DIRICHLET BC which makes the solution much different than a 0 albedo BC or a more accurate MC result or something
'''input_folder = 'simulations/ProblemData_1D_scaling_tests/'
n_dim = 1
min_qubits = 1
max_qubits = 7
max_dim_size = int(math.pow(2,max_qubits))
x_range = 3.7074
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = x_range / (data.n + 1)
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_FD_construct_L_F_matrices()
    #data.h = x_range / (data.n) # different h for finite volume and finite difference
    #A_matrix_finite_volume_test, B_matrix_finite_volume_test = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    fund_eigvec = np.kron(eigvecs[:,0].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # extend/interpolate eigenvectors onto finest grid and make first value positive
    eigenvector_results[i - min_qubits,:] = fund_eigvec / np.linalg.norm(fund_eigvec)

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''

############# 2D Diffusion FD eigenvalue results ###############
# Dimensions: 2
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Discretization Scheme: Finite Difference
# Notes: 
'''input_folder = 'simulations/ProblemData_2D_scaling_tests/'
n_dim = 2
min_qubits = 1
max_qubits = 5
max_dim_size = int(math.pow(2,max_qubits))
x_range = 4.0
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, int(math.pow(max_dim_size,n_dim))))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = x_range / (data.n + 1)
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_FD_construct_L_F_matrices()
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    fund_eigvec = np.kron(eigvecs[:,0].reshape(tuple([int(math.pow(2,i)) for d in range(n_dim)])),np.ones(tuple([int(max_dim_size/int(math.pow(2,i))) for d in range(n_dim)]))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # extend/interpolate eigenvectors onto finest grid and make first value positive
    eigenvector_results[i - min_qubits,:] = fund_eigvec / np.linalg.norm(fund_eigvec)

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])'''



############# Plot Results ###############

# plot eigenvectors
'''if n_dim == 1:
    for i in range(max_qubits - min_qubits + 1):
        ax = sns.heatmap(np.abs(eigenvector_results[i,:]).reshape(int(math.pow(2,max_qubits)), 1), linewidth=0.5)
        ax.invert_yaxis()
        plt.title("Expanded Eigenvector Solution")
        plt.figure()
if n_dim == 2:
    for i in range(max_qubits - min_qubits + 1):
        ax = sns.heatmap(np.abs(eigenvector_results[i,:]).reshape(int(math.pow(2,max_qubits)), int(math.pow(2,max_qubits))), linewidth=0.5)
        ax.invert_yaxis()
        plt.title("Actual Fine Solution")
        plt.figure()'''

# plot scaling of eigenvalues and eigenvectors
plt.plot(np.power(2,list(range(max_qubits)))[:-1], abs(eigenvalue_error[:-1]))
plt.title("error in eigenvalues")
plt.xlabel("number of finite volumes, N")
plt.ylabel("lambda eigenvalue error")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(eigenvalue_error[:-1]))
#C = (abs(eigenvalue_error[0])) 
C = 1
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], C * np.square(1/np.power(2,list(range(max_qubits))))[:-1])
plt.title("error in eigenvalues")
plt.xlabel("log of distance between finite volumes (h)")
plt.ylabel("log of lambda eigenvalue error")
plt.legend(['eigenvalue error', 'scaled ' + r'$h^2$'])
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(eigenvector_l2_norm[:-1]))
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(eigenvector_linf_norm[:-1]))
#C = (abs(eigenvalue_error[0])) 
C = 1
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], C * np.square(1/np.power(2,list(range(max_qubits))))[:-1]) # h^2 line
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], C * 1/np.power(2,list(range(max_qubits)))[:-1]) # h line
plt.title("error in eigenvectors")
plt.xlabel("log of distance between finite volumes (h)")
plt.ylabel("log of eignevector error")
plt.legend(['eigenvector l2 error', 'eigenvector l_inf error', 'scaled ' + r'$h^2$', 'scaled ' + r'$h$', 'scaled ' + r'$\sqrt{h}$'])
plt.figure()


# plot condition numbers
'''plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(condAs[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("A condition number")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(condBs[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("B condition number")
plt.figure()


# plot singular values
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(A_max_sings[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("A_norm")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(B_max_sings[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("B_norm")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(A_min_sings[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("A_inv_norm")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(B_min_sings[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("B_inv_norm")
plt.figure()'''



plt.show()