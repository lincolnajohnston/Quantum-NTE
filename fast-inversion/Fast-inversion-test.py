import sys
import os
sys.path.append(os.getcwd())
import ProblemData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian, eigh, svdvals, sqrtm, expm
import math

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Dirichlet BC where a = u_0, b = u_N
def get_laplacian_dirichlet_bc(N, x_range, a, b):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1) - np.diag(np.ones(N-2),-1))

    return_vec = np.zeros(N-1)
    return_vec[0] = a / (h*h)
    return_vec[N-2] = b / (h*h)

    return return_mat, return_vec

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Neumann BC where a = u_0, b = u_N
def get_laplacian_neumann_bc(N, x_range, a, b):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N+1)) - np.diag(np.ones(N),1) - np.diag(np.ones(N),-1))
    return_mat[0,0] = 1/(h*h)
    return_mat[N,N] = 1/(h*h)

    return_vec = np.zeros(N+1)
    return_vec[0] = -2 * a / h
    return_vec[N] = 2 * b / h

    return return_mat, return_vec

# return the A matrix and the b vector for the equation del^2(x) = 0. 1-D, Dirichlet BC where a = u_0, b = u_N
def get_laplacian_robin_bc(N, x_range, a, b, c, d):
    h = x_range / N
    return_mat = 1/(h*h) * (2 * np.diag(np.ones(N+1)) - np.diag(np.ones(N),1) - np.diag(np.ones(N),-1))
    return_mat[0,0] = 1/(h*h) * (1-a/h)
    return_mat[0,1] = 1/(h*h) * a/h
    return_mat[N,N] = 1/(h*h) * (1+c/h)
    return_mat[N,N-1] = 1/(h*h) * (-c/h)

    return_vec = np.zeros(N+1)
    return_vec[0] = b / (h*h)
    return_vec[N] = d / (h*h)

    return return_mat, return_vec

def get_discrete_cosine_transform(N, K_min, K_max):
    return_mat = np.zeros((K_max - K_min + 1,K_max - K_min + 1))
    #for j in range(1,len(return_mat)+1):
    #    for k in range(1, len(return_mat)+1):
    for j in range(K_min, K_max + 1):
        for k in range(K_min, K_max + 1):
            #return_mat[j-K_min,k-K_min] =  math.sqrt(2/N) * math.cos(math.pi * j * k / (N)) / (math.sqrt(2) if k % (N) == 0 else 1) / (2 if j % (N) == 0 else 1) # best DCT so far
            return_mat[j-K_min,k-K_min] =  math.sqrt(2/N) * math.cos(math.pi * j * k / (N)) / (math.sqrt(2) if j % (N) == 0 else 1) # testing
    return return_mat

def get_discrete_sine_transform(N, K_min, K_max):
    return_mat = np.zeros((K_max - K_min + 1,K_max - K_min + 1))
    #for j in range(1,len(return_mat)+1):
    #    for k in range(1, len(return_mat)+1):
    for j in range(K_min, K_max + 1):
        for k in range(K_min, K_max + 1):
            return_mat[j-K_min,k-K_min] = math.sqrt(2/N) * math.sin(math.pi * j * k / N) # correct values for the DST
            #return_mat[j-K_min,k-K_min] = math.sqrt(2/N) * math.sin(math.pi * j * k / N) / (math.sqrt(2) if k == K_min else 1)# / (math.sqrt(2) if j == K_min else 1)# testing random stuff
    return return_mat

def get_eigenvalues(N, K_min, K_max, x_range):
    h = x_range / N
    return np.diag([1/(h*h) * (2 - 2 * math.cos(math.pi * k / N)) for k in range(K_min, K_max + 1)])

# return True if matrix is unitary, False otherwise, O(len(matrix)^2)
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))

############# 1D Diffusion eigenvalue results ###############
# Dimensions: 1
# Equation Diffusion
# Geometry: Square Fuel Pin
# Notes: 
n_dim = 1
input_folder = 'simulations/Pu239_1G_1D_diffusion_fine/'
n_qubits = 3
N = int(math.pow(2,n_qubits))
input_file = 'input.txt'
x_range = 4

# create and modify input file
data = ProblemData.ProblemData(input_folder + input_file)
data.n = np.array([N] * n_dim)
data.h = x_range / data.n
data.initialize_BC()
data.initialize_geometry()
A_mat_size = math.prod(data.n) * data.G

##### Dirichlet B.C. Laplacian #####
a = 1 # Dirichlet BC parameter on left side
b = 2 # Dirichlet BC parameter on left side
dirichlet_laplacian, dirichlet_b = get_laplacian_dirichlet_bc(N, x_range, a, b)
dirichlet_eigvals, dirichlet_eigvecs = eigh(dirichlet_laplacian, eigvals_only=False)

# without the 0 and N nodes
sin_trans = get_discrete_sine_transform(N, 1, N-1)
print("sin_trans is a scaled unitary: ", is_unitary(sin_trans / np.linalg.norm(sin_trans,2)))
eigvals = get_eigenvalues(N, 1, N-1, x_range)
print("eigvals is a scaled unitary: ", is_unitary(eigvals / np.linalg.norm(eigvals,2)))

approx_lap_dir_matrix = sin_trans @ eigvals @ np.transpose(sin_trans)
unitary_lap_dir = approx_lap_dir_matrix / np.linalg.norm(approx_lap_dir_matrix, 2)
print("Desired Dirichlet Laplacian: ", dirichlet_laplacian)
print("Estimated Dirichlet Laplacian: ", np.round(approx_lap_dir_matrix,2))
print("Unitary gate for Dirichlet Laplacian: ", unitary_lap_dir)
print("Unitary gate is actually unitary: ", is_unitary(unitary_lap_dir))
print("Dirichlet b vector: ", dirichlet_b)

##### Neumann B.C. Laplacian #####
a = 1 # Neumann BC parameter on left side
b = 2 # neumann BC parameter on left side
neumann_laplacian, neumann_b = get_laplacian_neumann_bc(N, x_range, a, b)
neumann_eigvals, neumann_eigvecs = eigh(neumann_laplacian, eigvals_only=False)

cos_trans = get_discrete_cosine_transform(N, 0, N)
eigvals = get_eigenvalues(N, 0, N, x_range)

test1 = np.transpose(cos_trans)
test2 = eigvals @ np.transpose(cos_trans)

approx_lap_neu_matrix = np.transpose(cos_trans) @ eigvals @ cos_trans
print("Desired Neumann Laplacian: ", neumann_laplacian)
print("Estimated Neumann Laplacian: ", np.round(approx_lap_neu_matrix,2))
print("Neumann b vector: ", neumann_b)


##### Robin B.C. Laplacian #####
a = 2
b = 3
c = 4
d = 5
alpha = -(a/data.h + 1)
beta = 1

robin_laplacian, robin_b = get_laplacian_robin_bc(N, x_range, a, b, c, d)
robin_eigvals, robin_eigvecs = eigh(robin_laplacian, eigvals_only=False)

# with the 0 and N nodes
sin_trans = get_discrete_sine_transform(N+2, 1, N+1)
sin_trans_inv = np.linalg.inv(sin_trans)
sin_trans_transposed = np.transpose(sin_trans)
cos_trans = get_discrete_cosine_transform(N, 0, N)
cos_trans_inv = np.linalg.inv(cos_trans)
cos_trans_transposed = np.transpose(cos_trans)
eigvals_1 = get_eigenvalues(N, 0, N, x_range)
eigvals_2 = get_eigenvalues(N+2, 1, N+1, x_range)

# alpha times the matrix made for the Neumann BC + beta time the matrix made for the Dirichlet BC
# TODO: extend the Dirichlet matrix to be on the first and last rows and columns, this is currently broken because it is not
neu_term = cos_trans @ eigvals_1 @ cos_trans_transposed
dir_term = sin_trans @ eigvals_2 @ sin_trans_transposed
approx_lap_rob_matrix = alpha / (alpha + beta) * neu_term + beta / (alpha + beta) * (N*N) / ((N+2) * (N+2)) * dir_term
print("Desired Robin Laplacian: ", robin_laplacian)
print("Estimated Robin Laplacian: ", np.round(approx_lap_rob_matrix,2))
print("Error in Robin Laplacian: ", approx_lap_rob_matrix - robin_laplacian)
print("Robin b vector: ", robin_b)


# testing different laplacian matrices
cos_trans_test = get_discrete_cosine_transform(N, 0, N)
cos_trans_test_transposed = np.transpose(cos_trans)
eigvals_test = get_eigenvalues(N, 0, N, x_range)
neu_test_term = cos_trans_test @ eigvals_test @ cos_trans_test_transposed

A_matrix = data.diffusion_construct_diffusion_operator(A_mat_size)

eigvals, eigvecs = eigh(A_matrix, eigvals_only=False)



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
'''plt.plot(np.power(2,list(range(max_qubits)))[:-1], abs(eigenvalue_error[:-1]))
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
plt.figure()'''


# plot condition numbers
'''plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(condAs[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("A condition number")
plt.figure()

plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(condBs[:-1]), '-o')
plt.xlabel("h")
plt.ylabel("B condition number")
plt.figure()'''


# plot singular values
'''plt.loglog(1/np.power(2,list(range(max_qubits)))[:-1], abs(A_max_sings[:-1]), '-o')
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