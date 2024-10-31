import matplotlib.pyplot as plt
import os
import numpy as np

# read in precision numbers
sim_path = 'simulations/precision_vs_M_study/'
M_vals = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
precision_master_vec = []

for M in M_vals:
    i=0
    precision_vec = []
    while os.path.exists(sim_path + 'M=' + str(M) + '/saved_data/stats' + str(i) + '.txt'):
        f = open(sim_path + 'M=' + str(M) + '/saved_data/stats' + str(i) + '.txt', "r")
        line = f.readline().split()
        precision = float(line[1])
        f.close()
        precision_vec.append(1/precision)
        i += 1
    precision_master_vec.append(precision_vec)
#precision_master_vec = np.array(precision_master_vec)
precision_means = np.zeros(len(M_vals))
precision_stds = np.zeros(len(M_vals))
for i in range(len(M_vals)):
    precision_means[i] = np.mean(precision_master_vec[i])
    precision_stds[i] = np.std(precision_master_vec[i])
plt.errorbar(M_vals, precision_means, yerr=precision_stds, capsize=5)
#plt.xscale('log')
plt.xlabel("Number of timesteps")
plt.ylabel("Precision of AQC solution")
plt.show()