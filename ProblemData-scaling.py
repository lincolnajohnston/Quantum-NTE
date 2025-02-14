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
'''n_dim = 2
input_folder = 'simulations/ProblemData_2D_scaling_tests/'
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
    A_matrix, B_matrix = data.diffusion_construct_L_F_matrices(A_mat_size)
    eigvals, eigvecs = eigh(A_matrix, B_matrix, eigvals_only=False)
    eigenvalue_results[i - min_qubits] = eigvals[0]

closest_eig = eigenvalue_results[-1]
eigenvalue_error = eigenvalue_results - closest_eig

print("eigenvalues: ", eigenvalue_results)
print("eigenvalue errors: ", eigenvalue_error)'''


############# 1D Diffusion eigenvalue results ###############
'''input_folder = 'simulations/ProblemData_1D_scaling_tests/'
min_qubits = 1
max_qubits = 8
x_range = 4.0
#input_files = ['input-N=' + str(int(math.pow(2,i))) + '.txt' for i in range(1,max_qubits + 1)]
input_file = 'input.txt'

eigenvalue_results = np.zeros(max_qubits - min_qubits + 1)
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

print("eigenvalues: ", eigenvalue_results)

closest_eig = eigenvalue_results[-1]
eigenvalue_error = eigenvalue_results - closest_eig

print("eigenvalue errors: ", eigenvalue_error)'''


############# Plot Results ###############
plt.plot(np.power(2,list(range(max_qubits))), eigenvalue_results)
plt.title("fundamental eigenvalues")
plt.xlabel("N")
plt.ylabel("lambda eigenvalue")
plt.figure()

plt.plot(np.power(2,list(range(max_qubits))), abs(eigenvalue_error))
plt.title("error in eigenvalues")
plt.xlabel("number of finite volumes, N")
plt.ylabel("lambda eigenvalue error")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits))), abs(eigenvalue_error))
#C = (abs(eigenvalue_error[0])) 
C = 1
plt.loglog(1/np.power(2,list(range(max_qubits))), C * np.square(1/np.power(2,list(range(max_qubits)))))
plt.title("error in eigenvalues")
plt.xlabel("log of distance between finite volumes (h)")
plt.ylabel("log of lambda eigenvalue error")
plt.legend(['eigenvalue error', 'scaled ' + r'$h^2$'])
plt.show()