import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
import math

from qiskit import transpile
from qiskit_aer.aerprovider import QasmSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library import StatePreparation, CXGate, XGate, ZGate, QFT, HGate, RYGate, U1Gate, IntegerComparator, DraperQFTAdder
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Operator
import fable

# Returns the matrix that does a basis transform from the wavelet basis to the hat function nodal basis
def getL(n_min, n_max):
    N_max = int(math.pow(2,n_max)) # size of most fine wavelet set
    N_min = int(math.pow(2,n_min)) # size of most coarse wavelet set
    L = np.zeros((N_max-1,N_max-N_min)) # Basis transformation matrix from wavelet to nodal hat function basis

    s = 0 # leftmost row index in stencil
    j = 1 # jump between discrete points on wavelet stencil grid (in terms of number of points on finest grid)
    ref = 0 # current column
    for i in range(n_max-n_min):
        N_cur = int(N_max * math.pow(2,-i-1)) # size of wavelet set at the current level
        dilation_radius = int(2**i - 1) # radius of points at which the fine nodal basis is needed to represent the current wavelet
        dilation_list = 1/(2**i) * np.array(list(range(1,2**(i)+1,1)) + list(range(2**(i)-1,0,-1)))

        # special case for most coarse wavelet (only one point)
        if (n_min == 0 and i == n_max -1):
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s + 0*j + off,ref+0] += dilation_list[dil_index]
            continue
            

        # column 0 (left boundary):
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 0*j + off,ref+0] += dilation_list[dil_index] * 9/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 1*j + off,ref+0] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s + 2*j + off,ref+0] += dilation_list[dil_index] * 1/10

        # columns [1,N-2] (interior points):
        for l in range(1,N_cur-1):
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l-2) + off,ref+l] += dilation_list[dil_index] * 1/10
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l-1) + off,ref+l] += dilation_list[dil_index] * -3/5
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l) + off,ref+l] += dilation_list[dil_index] * 1
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l+1) + off,ref+l] += dilation_list[dil_index] * -3/5
            for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
                L[s+j*(2*l+2) + off,ref+l] += dilation_list[dil_index] * 1/10

        # column N-1 (right boundary):
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-4) + off,ref+N_cur-1] += dilation_list[dil_index] * 1/10
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-3) + off,ref+N_cur-1] += dilation_list[dil_index] * -3/5
        for dil_index, off in enumerate(range(-dilation_radius,dilation_radius+1)):
            L[s+j*(2*N_cur-2) + off,ref+N_cur-1] += dilation_list[dil_index] * 9/10

        s += j
        j *= 2 # double jump size for next (coarser) grid
        ref += N_cur

    return L

# get the matrix that maps the centers of the wavelet basis change matrix
# only has 1's in the matrix, boundary conditions and dilation not included
# if full=True, adds one extra row and column to make the matrix size a power of 2
def get_Lu(n, full=False):
    N = int(math.pow(2,n))
    Lu = np.zeros((N-1+int(full),N-1+int(full)))
    for i in range(N-1):
        section = n - int(math.log2(N-i-1)) - 1
        Lu[int(2**section - 1 + 2**(section+1)*(i-(2**section-1)*N/2**section)),i] = 1
    if full:
        Lu[N-1,N-1] = 1
    return Lu

def get_E(N, offset = 1):
    matrix = np.zeros((N, N))
    matrix[0:N,0:N] += np.diag(np.ones(N)) # diagonal terms

    matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=offset)  # k=1 for superdiagonal
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=N-offset)  # k=N-1 for superdiagonal

    matrix[0:N,0:N] += np.diag(0.5 * np.ones(N-offset), k=-offset) # k=-1 for subdiagonal
    matrix[0:N,0:N] += np.diag(0.5 * np.ones(offset), k=offset-N) # k=-1 for subdiagonal

    matrix[N-1,N-1] = 1
    return matrix

# modified version of E that is unitary
def get_E_mod(N):
    matrix = np.zeros((N, N))
    matrix += np.diag(2/math.sqrt(6) * np.ones(N))
    matrix += np.diag(1/math.sqrt(6) * np.ones(N-1), k=1)  # k=1 for superdiagonal
    matrix += np.diag(-1/math.sqrt(6) * np.ones(N-1), k=-1) # k=-1 for subdiagonal
    matrix[0,N-1] = -1/math.sqrt(6)
    matrix[N-1,0] = 1/math.sqrt(6)
    return matrix

# Apply the E operator of size (N-1 x N-1) to the quantum circuit, qc
# index_list is the qubits of qc to apply E to, c_index is a control qubit
# ancilla_1_index_list is the ancilla qubits to use for use in the DraperQFTAdder
# ancilla_2_index_list is the ancilla qubits to use for representing the amplitudes of the unitary matrices in LCU
def apply_E_operator(qc, n, E_index_list, ancilla_1_index_list, ancilla_2_index_list, c_index, offset=1):
    # set the state for the LCU linear combination
    N = int(2**n)
    ancilla_2_state = [1/math.sqrt(2), 1/2, 1/2, 0]
    ancilla_2_state_prep = StatePreparation(ancilla_2_state)
    qc.append(ancilla_2_state_prep, ancilla_2_index_list)

    # create a state in the ancilla_1 register to represent the integer 'offset'
    binary_offset = bin(offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # positive offset shift
    qc.x(ancilla_2_index_list[1])
    adder_plus = DraperQFTAdder(n, kind='fixed').control(3)
    qc.append(adder_plus,c_index + ancilla_2_index_list + ancilla_1_index_list + E_index_list)
    qc.x(ancilla_2_index_list[1])

    # uncompute the ancilla_1 register to all zeros
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # create a state in the ancilla_1 register to represent the integer N-offset, adding this is the same as subtracting offset
    binary_offset = bin(N-offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # negative offset shift
    qc.x(ancilla_2_index_list[0])
    adder_minus = DraperQFTAdder(n, kind='fixed').control(3)
    qc.append(adder_minus,c_index + ancilla_2_index_list + ancilla_1_index_list + E_index_list)
    qc.x(ancilla_2_index_list[0])

    # reset ancilla_1 register
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])
    
    phase_change_gate = ZGate().control(1, ctrl_state='0')
    #qc.append(phase_change_gate, ancilla_2_index_list[::-1])

    ancilla_2_state_prep_2_inv = StatePreparation(ancilla_2_state, inverse=True)
    qc.append(ancilla_2_state_prep_2_inv, ancilla_2_index_list)

    #qc.measure(ancilla_2_index_list, range(2)) # LCU block-encoding succeeds when this measurement is two zero states

# Apply the F operator of size (N-1 x N-1) to the quantum circuit, qc
# F_index_list is the qubits of qc to apply F to, c_index is a control qubit
# ancilla_1_index_list is the ancilla qubits to use for use in the DraperQFTAdder
# ancilla_2_index_list is the ancilla qubits to use for representing the amplitudes of the unitary matrices in LCU
def apply_F_operator(qc, n, F_index_list, ancilla_1_index_list, ancilla_2_index_list, c_index, offset=1):
    # set the state for the LCU linear combination
    N = int(2**n)
    ancilla_2_state = [1/math.sqrt(2.4), 1/2, 1/2, 1/math.sqrt(24), 1/math.sqrt(24), 0, 0, 0]
    ancilla_2_state_prep = StatePreparation(ancilla_2_state)
    qc.append(ancilla_2_state_prep, ancilla_2_index_list)

    # create a state in the ancilla_1 register to represent the integer 'offset'
    binary_offset = bin(offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # positive offset 0.6 shift
    qc.x(ancilla_2_index_list[1])
    qc.x(ancilla_2_index_list[2])
    adder_plus = DraperQFTAdder(n, kind='fixed', name='0.6 up shift').control(4)
    qc.append(adder_plus,c_index + ancilla_2_index_list + ancilla_1_index_list + F_index_list)
    qc.x(ancilla_2_index_list[1])
    qc.x(ancilla_2_index_list[2])

    # uncompute the ancilla_1 register to all zeros
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # create a state in the ancilla_1 register to represent the integer N-offset, adding this is the same as subtracting offset
    binary_offset = bin(N-offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # negative offset 0.6 shift
    qc.x(ancilla_2_index_list[0])
    qc.x(ancilla_2_index_list[2])
    adder_minus = DraperQFTAdder(n, kind='fixed', name='0.6 down shift').control(4)
    qc.append(adder_minus,c_index + ancilla_2_index_list + ancilla_1_index_list + F_index_list)
    qc.x(ancilla_2_index_list[0])
    qc.x(ancilla_2_index_list[2])

    # uncompute ancilla_1 register
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])
    
    # apply phase change to make the 0.6 values negative
    phase_change_gate = ZGate().control(2, ctrl_state='00')
    qc.append(phase_change_gate, [ancilla_2_index_list[i] for i in [-1,-2,-3]])

    phase_change_gate = ZGate().control(2, ctrl_state='00')
    qc.append(phase_change_gate, [ancilla_2_index_list[i] for i in [-1,-3,-2]])

    # create a state in the ancilla_1 register to represent the integer 'offset'
    binary_offset = bin(2*offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # positive offset 0.1 shift
    qc.x(ancilla_2_index_list[2])
    adder_plus = DraperQFTAdder(n, kind='fixed', name='0.1 up shift').control(4)
    qc.append(adder_plus,c_index + ancilla_2_index_list + ancilla_1_index_list + F_index_list)
    qc.x(ancilla_2_index_list[2])

    # uncompute the ancilla_1 register to all zeros
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # create a state in the ancilla_1 register to represent the integer N-offset, adding this is the same as subtracting offset
    binary_offset = bin(N-2*offset)  # binary of offset
    binary_offset_list = [int(digit) for digit in binary_offset[2:].zfill(n)]
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    # negative offset 0.1 shift
    qc.x(ancilla_2_index_list[0])
    qc.x(ancilla_2_index_list[1])
    adder_minus = DraperQFTAdder(n, kind='fixed', name='0.1 down shift').control(4)
    qc.append(adder_minus,c_index + ancilla_2_index_list + ancilla_1_index_list + F_index_list)
    qc.x(ancilla_2_index_list[0])
    qc.x(ancilla_2_index_list[1])

    # uncompute ancilla_1 register
    for b_i,b in enumerate(binary_offset_list):
        if b:
            qc.x(ancilla_1_index_list[-1-b_i])

    ancilla_2_state_prep_2_inv = StatePreparation(ancilla_2_state, inverse=True)
    qc.append(ancilla_2_state_prep_2_inv, ancilla_2_index_list)

n=3
N = int(2**n)
L = getL(0,n) # basis change from wavelet to hat function

# Test how to make the L matrix classically
Lu = get_Lu(n, full=True)
E1 = get_E(N, offset=1)
E2 = get_E(N, offset=2)
E4 = get_E(N, offset=4)
E = E1 #@ E2# @ E4
test = E @ Lu


x_vals = list(range(int(N/2),int(N/2)+1))
#x_vals = [1]

for x_val in x_vals:
    # b vector state preparation
    x_state = np.zeros(int(N))
    x_state[x_val] = 1
    #x_state = [1/math.sqrt(2),0,0,0,1/math.sqrt(2),0,0,0]
    #x_state = np.array([0,0,0,0,1.0,0,0,0])

    # set up the quantum circuit
    qc = QuantumCircuit(6*n+2,2)

    x_state_prep = StatePreparation(x_state)
    qc.append(x_state_prep, list(range(n)))


    # Use CNOTs to create the flag states for the E dilator
    for i in range(1,n):
        x_gate = XGate().control(n-i)
        qc.append(x_gate, list(range(n-1,i-1,-1)) + [4*(n)-1+i])

    # Use CNOTs to create the flag states for the F expansion
    for i in range(n):
        x_gate = XGate().control(i+1, ctrl_state='0'+'1'*i)
        qc.append(x_gate, list(range(n-1,n-2-i,-1)) + [5*(n)+2+i])

    LuGate = UnitaryGate(Lu, label="L_u Gate")
    qc.append(LuGate,list(range(n)))


    # switch flag states for testing:
    #qc.x(3*n+2)
    #qc.x(3*n+3)


    # apply the F gates
    for i in range(0,n):
        apply_F_operator(qc, n+1, list(range(n+1)), list(range(n+1,2*n+2)), list(range(5*n-1,5*n+2)), [5*n+2+i], offset=int(2**(i)))

    # apply the E gates
    for i in range(1,n):
        apply_E_operator(qc, n+1, list(range(n+1)), list(range(n+1,2*n+2)), list(range(2*(n+i-1)+2,2*(n+i)+2)), [4*n-1+i], offset=int(2**(n-i-1)))


    ##### reverse the flag bits, just for easier viewing of the statevector during testing #####
    # reverse E dilator flag qubits
    for i in range(1,n):
        if x_val >= N-2**i:
            qc.x([4*(n)-1+i])
    # reverse the F expansion matrix flag qubit
    qc.x(6*n+2-math.ceil(math.log2(N-x_val)))


    qc.save_statevector()

    # Run emulator
    backend = QasmSimulator(method="statevector")
    new_circuit = transpile(qc, backend)
    #print(dict(new_circuit.count_ops())) # print the counts of each type of gate
    job = backend.run(new_circuit)
    job_result = job.result()

    # print statevector of non-junk qubits
    state_vec = job_result.get_statevector(qc).data
    out_state = state_vec[:N]
    print("Input state: ", np.round(x_state,decimals=3))
    print("Output state: ", np.round(out_state, decimals=5), "\n")


qc.draw('mpl', filename="prewavelet-basis-change-test.png")

print("done")