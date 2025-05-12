import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Code Description: 
# Interpolates a coarse mesh represented as a quantum state over 'nc' qubits to a quantum state representing the same solution on a fine mesh over 'nf' qubits
# Uses 1-qubit gates to perform the inteprolation. Intention of this algorithm is to use in the FEEN/QPE algorithm to improve the overlap between the coarse mesh input
# and the fundamental eigenvector of the fine mesh matrix. Only uses numpy calculations instead of Qiskit emulation.


# basically makes the solution piecewise constant using each grid point on the coarse grid
def expand_grid_constant(coarse_sol, nc, nf):
    #H_gate = [[1, 1], [1, -1]]
    equal_state = [1/math.sqrt(2), 1/math.sqrt(2)]
    expanded_state = coarse_sol
    for i in range(nf-nc):
        expanded_state = np.kron(expanded_state, equal_state)
    return expanded_state

# Creates a quantum state where each term is approxmiately a sequential integer (and then scaled)
def get_interpolater_state(N):
    #return [i for i in range(int(math.pow(2,N)))] # the state we are aiming to approximate
    equal_state = [1/math.sqrt(2), 1/math.sqrt(2)]
    large_equal_state = [1]
    total_norm = 1
    out_state = [1]
    for i in range(N):
        q_n = [1,math.pow(N,1/(N))-math.pow(i,1/(N))+1]
        q_n_norm = np.linalg.norm(q_n)
        q_n = q_n / q_n_norm
        total_norm = total_norm / q_n_norm

        out_state = np.kron(out_state,q_n)
        large_equal_state = np.kron(large_equal_state, equal_state)
    large_equal_state = large_equal_state * math.pow(2,N/2) * total_norm
    int_state = out_state - large_equal_state
    return int_state / np.linalg.norm(int_state)

# tries to approximately linearly interpolate the coarse solution to get a fine solution
def expand_grid_linear_interpolate(coarse_sol, nc, nf):
    dn = nf-nc
    C_coarse = np.linalg.norm(coarse_sol)
    coarse_sol_state = coarse_sol / C_coarse

    diff_vec = coarse_sol[1:] - coarse_sol[:-1]
    diff_vec = np.append(diff_vec, 0)
    C_diff_vec = np.linalg.norm(diff_vec)
    diff_vec_state = diff_vec / C_diff_vec

    equal_state = [1/math.sqrt(2), 1/math.sqrt(2)]
    large_equal_state = equal_state
    for i in range(dn-1):
        large_equal_state = np.kron(large_equal_state, equal_state)
    interp_state = get_interpolater_state(dn)
    plt.plot(interp_state)
    plt.title("interpolation state")
    plt.figure()

    #interpolater_norm = math.sqrt(math.pow(2,dn)*(math.pow(2,dn) + 1)*(math.pow(2,dn+1)+1)/6) # theroetical norm for the perfect interpolator state
    interpolater_norm = (math.pow(2,dn)-1) / interp_state[-1] # norm that assures that the last value of interp_state is math.pow(2,dn)-1, which is directly on the linear interpolation line
    new_state = math.pow(2,dn/2) * C_coarse * (np.kron(coarse_sol_state,large_equal_state)) + interpolater_norm* C_diff_vec / math.pow(2,dn) * (np.kron(diff_vec_state, interp_state))

    return new_state

nc = 4
Nc = int(math.pow(2,nc))
nf = 7
Nf = int(math.pow(2,nf))

x_vals_c = np.array(range(Nc))
x_vals_f = np.array(range(Nf)) / (Nf/Nc)

coarse_sol = np.array([random.uniform(0, 1) for _ in range(Nc)])

fine_sol = expand_grid_linear_interpolate(coarse_sol, nc, nf)
plt.plot(x_vals_c, coarse_sol)
plt.plot(x_vals_f, fine_sol)
'''plt.figure()
plt.plot(x_vals_c,fine_sol[::int(Nf/Nc)] - coarse_sol)
plt.title("difference between fine and coarse solution (at coarse grid points)")'''
plt.show()


