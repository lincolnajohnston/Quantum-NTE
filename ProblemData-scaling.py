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
'''n_dim = 1
input_folder = 'simulations/ProblemData_1D_scaling_tests_fuel_pin/'
min_qubits = 1
max_qubits = 10
ranges = [4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    B_matrix, A_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[-1]

closest_eig = eigenvalue_results[-1]
eigenvalue_error = eigenvalue_results - closest_eig

print("eigenvalues: ", eigenvalue_results)
print("eigenvalue errors: ", eigenvalue_error)'''


############# 2D Diffusion eigenvalue results ###############
# Dimensions: 2
# Equation Diffusion
# Geometry: Square Fuel Pin
# Notes: 
'''n_dim = 2
input_folder = 'simulations/ProblemData_2D_scaling_tests_fuel_pin/'
min_qubits = 1
max_qubits = 5
ranges = [4.0, 4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))] * n_dim)
    data.h = ranges / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    B_matrix, A_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[-1]

closest_eig = eigenvalue_results[-1]
eigenvalue_error = eigenvalue_results - closest_eig

print("eigenvalues: ", eigenvalue_results)
print("eigenvalue errors: ", eigenvalue_error)'''



############# 3D Diffusion eigenvalue results ###############
# Dimensions: 3
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: still shows the h^2 dependence like in the 1D case and the same as the local truncation error
'''n_dim = 3
input_folder = 'simulations/ProblemData_3D_scaling_tests/'
min_qubits = 1
max_qubits = 4
ranges = [4.0, 4.0, 4.0] # the plotting assumes that the ranges in each dimension are all the same
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
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
    
closest_eig = eigenvalue_results[-1]
eigenvalue_error = eigenvalue_results - closest_eig

print("eigenvalues: ", eigenvalue_results)
print("eigenvalue errors: ", eigenvalue_error)'''


############# 2D Diffusion eigenvalue results ###############
# Dimensions: 2
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: still shows the h^2 dependence like in the 1D case and the same as the local truncation error
n_dim = 2
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
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,0].reshape((int(math.pow(2,i)), int(math.pow(2,i)))),np.ones((int(max_dim_size/int(math.pow(2,i))), int(max_dim_size/int(math.pow(2,i)))))).flatten() * eigvecs[0,0] / abs(eigvecs[0,0]) # scale eigenvector and make first value positive

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_diffs = [eigenvector_results[i,:] - closest_eigenvector for i in range(max_qubits - min_qubits + 1)]
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])


############# 1D Diffusion eigenvalue results ###############
# Dimensions: 1
# Equation Diffusion
# Geometry: Homogeneous Fuel
# Notes: shows h^2 dependence of eigenvalue error, same as the local truncation error
'''input_folder = 'simulations/ProblemData_1D_scaling_tests/'
n_dim = 1
min_qubits = 1
max_qubits = 8
max_A_mat_size = int(math.pow(2,max_qubits * n_dim))
x_range = 4.0
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
eigenvector_results = np.zeros((max_qubits - min_qubits + 1, max_A_mat_size))
data = ProblemData.ProblemData(input_folder + input_file)
for i in range(min_qubits, max_qubits + 1):
    data.n = np.array([int(math.pow(2,i))])
    data.h = x_range / data.n
    data.initialize_BC()
    data.initialize_geometry()
    A_mat_size = math.prod(data.n) * data.G
    A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]
    eigenvector_results[i - min_qubits,:] = np.kron(eigvecs[:,0],np.ones(int(max_A_mat_size/A_mat_size))) * eigvecs[0,0] / abs(eigvecs[0,0]) # scale eigenvector and make first value positive

print("eigenvalues: ", eigenvalue_results)

closest_eig = eigenvalue_results[-1]
closest_eigenvector = eigenvector_results[-1,:]

eigenvalue_error = eigenvalue_results - closest_eig
#eigenvector_error = eigenvector_results - closest_eigenvector
eigenvector_l2_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector) for i in range(max_qubits - min_qubits + 1)])
eigenvector_linf_norm = np.array([np.linalg.norm(eigenvector_results[i,:] - closest_eigenvector, ord=np.inf) for i in range(max_qubits - min_qubits + 1)])

print("eigenvalue errors: ", eigenvalue_error)'''


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
plt.plot(np.power(2,list(range(max_qubits)))[:-1], eigenvalue_results[:-1])
plt.title("fundamental eigenvalues")
plt.xlabel("N")
plt.ylabel("lambda eigenvalue")
plt.figure()

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
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], C * np.square(1/np.power(2,list(range(max_qubits))))[:-1])
plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], C * 1/np.power(2,list(range(max_qubits)))[:-1])
plt.title("error in eigenvectors")
plt.xlabel("log of distance between finite volumes (h)")
plt.ylabel("log of eignevector error")
plt.legend(['eigenvector l2 error', 'eigenvector l_inf error', 'scaled ' + r'$h^2$', 'scaled ' + r'$h$'])
plt.show()