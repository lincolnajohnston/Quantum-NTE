import numpy as np

class Material:
    def __init__(self, mat_name, xs_file, G, n_x, n_y):
        self.sigma_a = np.zeros(G)
        self.nu_sigma_f = np.zeros(G)
        self.D = np.zeros(G)
        self.Q = np.zeros(G)
        #for sp3 only below
        self.sigma_t = np.zeros(G)
        self.sigma_s0 = np.zeros(G)
        self.sigma_s2 = np.zeros(G)
        self.sigma_sgg = np.zeros((G,G)) # first index is incoming group, second index is outgoing group
        self.chi = np.zeros(G)
        self.D2 = np.zeros(G)
    

        file = open(xs_file, "r")
        xs_data = np.zeros((G,6+2*G))
        for g in range(G):
            xs_data[g,:] = [float(numeric_string) for numeric_string in file.readline().split()]
        for g_p in range(G):
            self.sigma_t[g_p] = xs_data[g_p,1] 
            self.sigma_a[g_p] = xs_data[g_p,1] - xs_data[g_p,2]
            self.nu_sigma_f[g_p] = xs_data[g_p,3]
            self.sigma_s0[g_p] = xs_data[g_p,2]
            for g in range(G):
                self.sigma_sgg[g_p, g] = xs_data[g_p,4+g]
            self.chi[g_p] = xs_data[g_p,4+2*G]
            self.D[g_p] = 1 / (3 * (xs_data[g_p,1] - np.sum(xs_data[g_p,4 + G:4 + 2*G]))) 
            self.D2[g_p] = 1 / (3 * (xs_data[g_p,1] - np.sum(np.zeros(8)))) # TODO: Add 3rd oder scattering cross sections in to the data file so D2 can be created, sigma_s3 is set to 0 temporarily here
            self.Q[g_p] = xs_data[g_p,5+2*G]
        print("loaded cross sections")