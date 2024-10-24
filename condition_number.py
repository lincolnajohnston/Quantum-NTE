import numpy as np
from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import ishermitian
import time
import ProblemData
import LcuFunctions

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
np.set_printoptions(threshold=np.inf)

filenames = ["fuel", "fuel_no_fission"]
condition_numbers = []

#check if matrix is unitary
def is_unitary(A_matrix):
    A_conj_transpose = np.conj(A_matrix).T
    identity_matrix = np.eye(len(A_matrix))
    #calculate At* A to see if it's approximately identity matrix
    unitary_check = np.dot(A_conj_transpose, A_matrix)
    return np.allclose(unitary_check, identity_matrix, atol=1e-10) #tolerance

for filename in filenames:
    material = {
        "fuel": filename,
        "moderator": "water"
    }

    data = ProblemData.ProblemData("input.txt", material)

    # make A matrix and b vector
    if data.sim_method == "sp3":
        A_mat_size = 2 * (data.n_x) * (data.n_y) * data.G
        A_matrix, b_vector = data.sp3_construct_A_matrix(A_mat_size) 
    elif data.sim_method == "diffusion":
        A_mat_size = (data.n_x) * (data.n_y) * data.G
        A_matrix, b_vector = data.diffusion_construct_A_matrix(A_mat_size)
        

    # Do LCU routine (https://arxiv.org/pdf/1511.02306.pdf), equation 18
    num_LCU_bits = 4
    num_unitaries = pow(2,num_LCU_bits)
    last_error_norm = np.inf

    A_mat_size = len(A_matrix)
    if(not ishermitian(A_matrix)): # make sure the matrix is hermitian
        quantum_mat = np.zeros((2*A_mat_size,2*A_mat_size))
        quantum_mat[A_mat_size:2*A_mat_size, 0:A_mat_size] = np.conj(A_matrix).T
        quantum_mat[0:A_mat_size, A_mat_size:2*A_mat_size] = A_matrix
        quantum_b_vector = np.zeros(2*len(b_vector))
        quantum_b_vector[0:len(b_vector)] = b_vector
        quantum_b_vector[len(b_vector):2*len(b_vector)] = b_vector
        A_mat_size *= 2
    else:
        quantum_mat = A_matrix
        quantum_b_vector = b_vector

    #check if quantum_mat is unitary
    if(is_unitary(quantum_mat)):
        print("Matrix is unitary.")
    else:
        print("Matrix is not unitary.")

    eigenvalues, eigenvectors = np.linalg.eig(quantum_mat)
    maxeigen = max(eigenvalues)
    mineigen = min(eigenvalues)
    print("max: ", maxeigen, "min", mineigen)
    condition_number = np.abs(max(eigenvalues) / min(eigenvalues))
    condition_numbers.append(condition_number)
    print(f"A condition number for {filename}: ", condition_number)

# Plotting the condition number vs absorption ratio
plt.figure(figsize=(10, 6))
plt.plot(filenames, condition_numbers, marker='o')
plt.xlabel('Absorption Ratio')
plt.ylabel('Condition Number')
plt.title('Condition Number vs Absorption Ratio')
plt.grid(True)
plt.savefig('condition_number_plot.png')

# {fuel, fuel_no_fission} {n_x, n_y} {delta_x = 0.5, delta_y = 0.5}

# {0.9999999999999966, 1.0000000000000029} {15, 15}
# {1.0000000000000009, 1.0000000000000044} {10, 10}
# {0.9999999999999972, 1.0000000000000053} {8, 8} {0.5, 0.5}
# {0.999999999999999,  0.9999999999999913} {8, 8} {0.25, 0.25}
# {1.0000000000000022, 1.000000000000001} {6, 6}
# {0.9999999999999992, 1.000000000000001} {3, 3}
# {0.9999999999999996, 0.9999999999999996} {1, 1}
#