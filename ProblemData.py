import numpy as np
import math



class ProblemData:
    def __init__(self, input_file):
        a = 1 # placeholder for now

    def read_input(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        self.n_x = int(lines[0].strip())
        self.n_pts_x = self.n_x + 2
        self.n_y = int(lines[1].strip())
        self.n_pts_y = self.n_y + 2
        self.delta_x = float(lines[2].strip())
        self.delta_y = float(lines[3].strip())
        self.sim_method = lines[4].strip()
    
    def initialize_BC(self):
        if self.sim_method == "sp3":
            top_I_3 = 0
            bottom_I_3 = 0
            right_I_3 = 0
            left_I_3 = 0
            self.top_x_I_3 = np.ones(self.n_pts_x) * top_I_3
            self.bottom_x_I_3 = np.zeros(self.n_pts_x) * bottom_I_3
            self.right_y_I_3 = np.zeros(self.n_pts_y) * right_I_3
            self.left_y_I_3 = np.zeros(self.n_pts_y) * left_I_3

            top_I_1 = 0
            bottom_I_1 = 0
            right_I_1 = 0
            left_I_1 = 0
            self.top_x_I_1 = np.ones(self.n_pts_x) * top_I_1
            self.bottom_x_I_1 = np.zeros(self.n_pts_x) * bottom_I_1
            self.right_y_I_1 = np.zeros(self.n_pts_y) * right_I_1
            self.left_y_I_1 = np.zeros(self.n_pts_y) * left_I_1
        elif self.sim_method == "diffusion":
            self.beta = 0.5 # related to albedo constant

    # return B.C.s at edge of problem domain
    def get_I_1_value(self, index):
        i = index[0]
        j = index[1]
        if (i == 0):
            return self.left_y_I_1[j]
        if (i == self.n_x-1):
            return self.right_y_I_1[j]
        if (j == 0):
            return self.bottom_x_I_1[i]
        if (j == self.n_y-1):
            return self.top_x_I_1[i]
        raise Exception("tried to get BC on non-boundary node")
    
    # return B.C.s at edge of problem domain
    def get_I_3_value(self, index):
        i = index[0]
        j = index[1]
        if (i == 0):
            return self.left_y_I_3[j]
        if (i == self.n_x-1):
            return self.right_y_I_3[j]
        if (j == 0):
            return self.bottom_x_I_3[i]
        if (j == self.n_y-1):
            return self.top_x_I_3[i]
        raise Exception("tried to get BC on non-boundary node")

    # get averaged diffusion coefficient in either the "x" or "y" direction for interior points
    # lower_index is lower index in the direction of the averaged diffusion coefficient
    # set_index is the other dimension index
    def get_av_D(self, direction, lower_index, set_index):
        if direction == "x":
            D_lower = self.D[unroll_index([lower_index, set_index], self.n_y)]
            D_upper = self.D[unroll_index([lower_index+1, set_index], self.n_y)]
            delta = self.delta_x
        elif direction == "y":
            D_lower = self.D[unroll_index([set_index, lower_index], self.n_y)]
            D_upper = self.D[unroll_index([set_index, lower_index+1], self.n_y)]
            delta = self.delta_y
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_av_D2(self, direction, lower_index, set_index):
        if direction == "x":
            D_lower = self.D2[unroll_index([lower_index, set_index], self.n_y)]
            D_upper = self.D2[unroll_index([lower_index+1, set_index], self.n_y)]
            delta = self.delta_x
        elif direction == "y":
            D_lower = self.D2[unroll_index([set_index, lower_index], self.n_y)]
            D_upper = self.D2[unroll_index([set_index, lower_index+1], self.n_y)]
            delta = self.delta_y
        return 2 * (D_lower/delta) * (D_upper/delta) / (D_lower/delta + D_upper/delta)


    def get_edge_D(self, x_i, y_i, delta):
        return 2 * (self.beta/2) * (self.D[unroll_index([x_i, y_i], self.n_y)]/delta) / (self.beta/2 + (self.D[unroll_index([x_i, y_i], self.n_y)]/delta))
        # Set material data at each finite difference point, O(N)
    

    def initialize_XSs(self):
        x_range = (self.n_pts_x - 1) * self.delta_x
        y_range = (self.n_pts_y - 1) * self.delta_y

        self.sigma_a = np.zeros(self.n_x*self.n_y)
        self.nu_sigma_f = np.zeros(self.n_x*self.n_y)
        self.D = np.zeros(self.n_x*self.n_y)
        self.Q = np.zeros(self.n_x*self.n_y)
        #for sp3 only below
        self.sigma_t = np.zeros(self.n_x*self.n_y)
        self.sigma_s0 = np.zeros(self.n_x*self.n_y)
        self.sigma_s2 = np.zeros(self.n_x*self.n_y)
        self.D2 = np.zeros(self.n_x*self.n_y)

        fuel_radius = min(x_range,y_range)/8
        #fuel_radius = 9999

        for i in range(self.n_x):
            for j in range(self.n_y):
                x_val = (i + 1) * self.delta_x - x_range/2
                y_val = (j + 1) * self.delta_y - y_range/2

                # fuel at center
                '''if (math.sqrt(x_val * x_val + y_val * y_val) < fuel_radius):
                    # use fuel XSs
                    self.sigma_a[i * self.n_y + j] = 1
                    self.nu_sigma_f[i * self.n_y + j] = 3
                    self.D[i * self.n_y + j] = 1
                    self.Q[i * self.n_y + j] = 5
                    #for sp3 only below
                    self.sigma_t[i * self.n_y + j] = 5
                    self.sigma_s0[i * self.n_y + j] = 4
                    self.sigma_s2[i * self.n_y + j] = 0.1
                    self.D2[i * self.n_y + j] = 2
                    
                else:
                    # use moderator XSs
                    self.sigma_a[i * self.n_y + j] = 2
                    self.D[i * self.n_y + j] = 1
                    #for sp3 only below
                    self.sigma_t[i * self.n_y + j] = 5
                    self.sigma_s0[i * self.n_y + j] = 3
                    self.sigma_s2[i * self.n_y + j] = 1
                    self.D2[i * self.n_y + j] = 2'''

                # 4 fuel pins
                if (math.sqrt(math.pow(abs(x_val)-x_range/4,2) + math.pow(abs(y_val)-y_range/4,2)) < fuel_radius):
                    # use fuel XSs
                    self.sigma_a[i * self.n_y + j] = 1
                    self.D[i * self.n_y + j] = 1
                    self.Q[i * self.n_y + j] = 5
                    self.nu_sigma_f[i * self.n_y + j] = 0 #sp3 looks better when this is 0
                    #for sp3 only below
                    self.sigma_t[i * self.n_y + j] = 5
                    self.sigma_s0[i * self.n_y + j] = 4
                    self.sigma_s2[i * self.n_y + j] = 0.1
                    self.D2[i * self.n_y + j] = 2
                    
                else:
                    # use moderator XSs
                    self.sigma_a[i * self.n_y + j] = 1
                    self.D[i * self.n_y + j] = 1
                    #for sp3 only below
                    self.sigma_t[i * self.n_y + j] = 5
                    self.sigma_s0[i * self.n_y + j] = 4
                    self.sigma_s2[i * self.n_y + j] = 2 
                    self.D2[i * self.n_y + j] = 2
                
    # use finite volume method to construct the A matrix representing the diffusion equation in the form Ax=b, O(N)
    def diffusion_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        for x_i in range(self.n_x):
            for y_i in range(self.n_y):
                i = unroll_index([x_i, y_i], self.n_y)
                if(x_i == 0): # left BC, normal vector = (-1,0)
                    J_x_minus = self.get_edge_D(x_i ,y_i,self.delta_x) * self.delta_y
                else:
                    J_x_minus = self.get_av_D("x",x_i-1,y_i) * self.delta_y
                    A_matrix[i,unroll_index([x_i-1, y_i], self.n_y)] =  -J_x_minus # (i-1,j) terms
                if(x_i == self.n_x - 1): # right BC, normal vector = (1,0)
                    J_x_plus = self.get_edge_D(x_i ,y_i,self.delta_x) * self.delta_y
                else:
                    J_x_plus = self.get_av_D("x",x_i,y_i) * self.delta_y
                    A_matrix[i,unroll_index([x_i+1, y_i], self.n_y)] =  -J_x_plus # (i+1,j) terms
                if(y_i == 0): # bottom BC, normal vector = (0,-1)
                    J_y_minus = self.get_edge_D(x_i ,y_i,self.delta_y)* self.delta_x
                else:
                    J_y_minus = self.get_av_D("y",y_i-1,x_i) * self.delta_x
                    A_matrix[i,unroll_index([x_i, y_i-1], self.n_y)] =  -J_y_minus # (i,j-1) terms
                if(y_i == self.n_y - 1): # right BC, normal vector = (0,1)
                    J_y_plus = self.get_edge_D(x_i ,y_i,self.delta_y) * self.delta_x
                else:
                    J_y_plus = self.get_av_D("x",y_i,x_i) * self.delta_x
                    A_matrix[i,unroll_index([x_i, y_i+1], self.n_y)] =  -J_y_plus # (i,j+1) terms
                A_matrix[i,i] = J_x_minus + J_x_plus + J_y_minus + J_y_plus + (self.sigma_a[i] - self.nu_sigma_f[i]) * self.delta_x * self.delta_y
        b_vector = self.Q * self.delta_x * self.delta_y
        return A_matrix, b_vector

    def sp3_construct_A_matrix(self, A_mat_size):
        fd_order = 2
        beta = 0.5
        phi_2_offset = self.n_x * self.n_y
        A_matrix = np.zeros((A_mat_size, A_mat_size))
        b_vector = np.zeros((A_mat_size))
        for x_i in range(self.n_x):
            for y_i in range(self.n_y):
                i = unroll_index([x_i, y_i], self.n_y)
                if(x_i == 0): # left BC, normal vector = (-1,0)
                    # these coefficients will be the same for all cells with the same material and mesh size so
                    # need to imporove efficiency of this by storing these values beforehand instead of recalculating for each B.C. cell
                    a1 = (1 + 4 * self.D[unroll_index([x_i, y_i], self.n_y)]/self.delta_x)
                    a2 = (-3/4) * self.D[unroll_index([x_i, y_i], self.n_y)]/self.D2[unroll_index([x_i, y_i], self.n_y)]
                    a3 = 2 * self.D[unroll_index([x_i, y_i], self.n_y)]/self.delta_x
                    a4 = (-3/4) * 2 * self.D[unroll_index([x_i, y_i], self.n_y)] / self.delta_x
                    a5 =  4 * self.D[unroll_index([x_i, y_i], self.n_y)]/self.delta_x * 2 * self.get_I_1_value([x_i,y_i])

                    b2 = (1 + (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.delta_x)
                    b1 = (-1/7) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.D[unroll_index([x_i, y_i], self.n_y)]
                    b4 = 2 * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.delta_x
                    b3 = (-2/7) * self.D2[unroll_index([x_i, y_i], self.n_y)] / self.delta_x
                    b5 =  (6/5) * (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.delta_x * self.get_I_3_value([x_i,y_i])

                    denom = (a1 - a2 * b1 / b2)
                    c1 = (a5 - a2 * b5 / b2) / denom
                    c2 = (a2 * b3 / b2 - a3) / denom
                    c3 = (a2 * b4 / b2 - a4) / denom

                    b_vector[i] += c1 *  self.delta_y
                    A_matrix[i,i] -= c2 *  self.delta_y
                    A_matrix[i,i+phi_2_offset] -= c3 *  self.delta_y

                    b_vector[i + phi_2_offset] += (b5 - b1 * c1) / b2 *  self.delta_y
                    A_matrix[i+phi_2_offset,i] -= (-b1 * c2 - b3) / b2 *  self.delta_y
                    A_matrix[i+phi_2_offset,i+phi_2_offset] -= (-b1 * c3 - b4) / b2 *  self.delta_y
                else:
                    # Phi_0 equations
                    x_minus_term_0 = self.get_av_D("x",x_i-1,y_i) *  self.delta_y
                    A_matrix[i,unroll_index([x_i-1, y_i], self.n_y)] =  -x_minus_term_0 # phi_0, (i-1,j) term
                    A_matrix[i,i] +=  x_minus_term_0 # phi_0, (i,j) term

                    # Phi_2 equations
                    x_minus_term_2 = self.get_av_D2("x",x_i-1,y_i) *  self.delta_y
                    A_matrix[i+phi_2_offset,unroll_index([x_i-1, y_i], self.n_y) + phi_2_offset] =  -x_minus_term_2 # phi_2, (i-1,j) term
                    A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_minus_term_2 # phi_2, (i,j) term
                if(x_i == self.n_x - 1): # right BC, normal vector = (1,0)
                    d1 = (-1 - 4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x)
                    d2 = (3/4) * self.D[unroll_index([x_i, y_i], self.n_y)]/self.D2[unroll_index([x_i, y_i], self.n_y)]
                    d3 = 2 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x
                    d4 = (-3/4) * 2 * self.D[unroll_index([x_i, y_i], self.n_y)] /  self.delta_x
                    d5 =  4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x * 2 * self.get_I_1_value([x_i,y_i])

                    e2 = (-1 - (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x)
                    e1 = (1/7) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.D[unroll_index([x_i, y_i], self.n_y)]
                    e4 = 2 * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x
                    e3 = (-2/7) * self.D2[unroll_index([x_i, y_i], self.n_y)] /  self.delta_x
                    e5 =  (6/5) * (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_x * self.get_I_3_value([x_i,y_i])

                    denom = (d1 - d2 * e1 / e2)
                    f1 = (d5 - d2 * e5 / e2) / denom
                    f2 = (d2 * e3 / e2 - d3) / denom
                    f3 = (d2 * e4 / e2 - d4) / denom

                    b_vector[i] -= f1 *  self.delta_y
                    A_matrix[i,i] += f2 *  self.delta_y
                    A_matrix[i,i+phi_2_offset] += f3 *  self.delta_y

                    b_vector[i + phi_2_offset] -= (e5 - e1 * f1) / e2 *  self.delta_y
                    A_matrix[i+phi_2_offset,i] += (-e1 * f2 - e3) / e2 *  self.delta_y
                    A_matrix[i+phi_2_offset,i+phi_2_offset] += (-e1 * f3 - e4) / e2 *  self.delta_y
                else:
                    # Phi_0 equations
                    x_plus_term_0 = self.get_av_D("x",x_i,y_i) *  self.delta_y
                    A_matrix[i,unroll_index([x_i+1, y_i], self.n_y)] =  -x_plus_term_0 # phi_0, (i+1,j) term
                    A_matrix[i,i] +=  x_plus_term_0 # phi_0, (i,j) term

                    # Phi_2 equations
                    x_plus_term_2 = self.get_av_D2("x",x_i,y_i) *  self.delta_y
                    A_matrix[i+phi_2_offset,unroll_index([x_i+1, y_i], self.n_y) + phi_2_offset] =  -x_plus_term_2 # phi_2, (i+1,j) term
                    A_matrix[i+phi_2_offset,i+phi_2_offset] +=  x_plus_term_2 # phi_2, (i,j) term
                if(y_i == 0): # bottom BC, normal vector = (0,-1)
                    a1 = (1 + 4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y)
                    a2 = (-3/4) * self.D[unroll_index([x_i, y_i], self.n_y)]/self.D2[unroll_index([x_i, y_i], self.n_y)]
                    a3 = 2 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y
                    a4 = (-3/4) * 2 * self.D[unroll_index([x_i, y_i], self.n_y)] /  self.delta_y
                    a5 =  4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y * 2 * self.get_I_1_value([x_i,y_i])

                    b2 = (1 + (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y)
                    b1 = (-1/7) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.D[unroll_index([x_i, y_i], self.n_y)]
                    b4 = 2 * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y
                    b3 = (-2/7) * self.D2[unroll_index([x_i, y_i], self.n_y)] /  self.delta_y
                    b5 =  (6/5) * (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y * 2 * self.get_I_3_value([x_i,y_i])

                    denom = (a1 - a2 * b1 / b2)
                    c1 = (a5 - a2 * b5 / b2) / denom
                    c2 = (a2 * b3 / b2 - a3) / denom
                    c3 = (a2 * b4 / b2 - a4) / denom

                    b_vector[i] += c1 *  self.delta_x
                    A_matrix[i,i] -= c2 *  self.delta_x
                    A_matrix[i,i+phi_2_offset] -= c3 *  self.delta_x

                    b_vector[i + phi_2_offset] += (b5 - b1 * c1) / b2 *  self.delta_x
                    A_matrix[i+phi_2_offset,i] -= (-b1 * c2 - b3) / b2 *  self.delta_x
                    A_matrix[i+phi_2_offset,i+phi_2_offset] -= (-b1 * c3 - b4) / b2 *  self.delta_x
                else:
                    # Phi_0 equations
                    y_minus_term_0 = self.get_av_D("y",x_i,y_i-1) *  self.delta_x
                    A_matrix[i,unroll_index([x_i, y_i-1], self.n_y)] =  -y_minus_term_0 # phi_0, (i,j-1) term
                    A_matrix[i,i] +=  y_minus_term_0 # phi_0, (i,j) term

                    # Phi_2 equations
                    y_minus_term_2 = self.get_av_D2("y",x_i,y_i-1) *  self.delta_x
                    A_matrix[i+phi_2_offset,unroll_index([x_i, y_i-1], self.n_y) + phi_2_offset] =  -y_minus_term_2 # phi_2, (i,j-1) term
                    A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_minus_term_2 # phi_2, (i,j) term
                if(y_i == self.n_y - 1): # right BC, normal vector = (0,1)
                    d1 = (-1 - 4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y)
                    d2 = (3/4) * self.D[unroll_index([x_i, y_i], self.n_y)]/self.D2[unroll_index([x_i, y_i], self.n_y)]
                    d3 = 2 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y
                    d4 = (-3/4) * 2 * self.D[unroll_index([x_i, y_i], self.n_y)] /  self.delta_y
                    d5 =  4 * self.D[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y * 2 * self.get_I_1_value([x_i,y_i])

                    e2 = (-1 - (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y)
                    e1 = (1/7) * self.D2[unroll_index([x_i, y_i], self.n_y)]/self.D[unroll_index([x_i, y_i], self.n_y)]
                    e4 = 2 * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y
                    e3 = (-2/7) * self.D2[unroll_index([x_i, y_i], self.n_y)] /  self.delta_y
                    e5 =  (6/5) * (80/21) * self.D2[unroll_index([x_i, y_i], self.n_y)]/ self.delta_y * self.get_I_3_value([x_i,y_i])

                    denom = (d1 - d2 * e1 / e2)
                    f1 = (d5 - d2 * e5 / e2) / denom
                    f2 = (d2 * e3 / e2 - d3) / denom
                    f3 = (d2 * e4 / e2 - d4) / denom

                    b_vector[i] -= f1 *  self.delta_x
                    A_matrix[i,i] += f2 *  self.delta_x
                    A_matrix[i,i+phi_2_offset] += f3 *  self.delta_x

                    b_vector[i + phi_2_offset] -= (e5 - e1 * f1) / e2 *  self.delta_x
                    A_matrix[i+phi_2_offset,i] += (-e1 * f2 - e3) / e2 *  self.delta_x
                    A_matrix[i+phi_2_offset,i+phi_2_offset] += (-e1 * f3 - e4) / e2 *  self.delta_x
                else:
                    # Phi_0 equations
                    y_plus_term_0 = self.get_av_D("y",x_i,y_i) *  self.delta_x
                    A_matrix[i,unroll_index([x_i, y_i+1], self.n_y)] =  -y_plus_term_0 # phi_0, (i,j+1) term
                    A_matrix[i,i] +=  y_plus_term_0 # phi_0, (i,j) term

                    # Phi_2 equations
                    y_plus_term_2 = self.get_av_D2("y",x_i,y_i) *  self.delta_x
                    A_matrix[i+phi_2_offset,unroll_index([x_i, y_i+1], self.n_y) + phi_2_offset] =  -y_plus_term_2 # phi_2, (i,j+1) term
                    A_matrix[i+phi_2_offset,i+phi_2_offset] +=  y_plus_term_2 # phi_2, (i,j) term
                A_matrix[i,i] += (self.sigma_t[i] - self.nu_sigma_f[i] - self.sigma_s0[i]) *  self.delta_x *  self.delta_y
                A_matrix[i,i + phi_2_offset] += (-2) * (self.sigma_t[i] - self.nu_sigma_f[i] - self.sigma_s0[i]) *  self.delta_x *  self.delta_y
                b_vector[i] += self.Q[i] *  self.delta_x *  self.delta_y

                A_matrix[i+phi_2_offset,i] += (-2/5) * (self.sigma_t[i] - self.nu_sigma_f[i] - self.sigma_s0[i]) *  self.delta_x *  self.delta_y
                A_matrix[i+phi_2_offset,i + phi_2_offset] += ((self.sigma_t[i] - self.nu_sigma_f[i] - self.sigma_s2[i]) + (4/5) * (self.sigma_t[i] - self.nu_sigma_f[i] - self.sigma_s0[i])) *  self.delta_x *  self.delta_y
                b_vector[i+phi_2_offset] += (-2/5) * self.Q[i] *  self.delta_x *  self.delta_y
        return A_matrix, b_vector



# convert 2D (x,y) index to 1D index
def unroll_index(index_vec, n_y):
    return index_vec[0]*n_y + index_vec[1]

# convert 1D index to 2D (x,y) index
def roll_index(index, n_y):
    return np.array([math.floor(index/n_y), index % n_y])


# old function to set BC flux in diffusion problems instead of using an albedo BC
'''def get_BC_value(index, n_x, n_y, left_y_BCs, right_y_BCs, bottom_x_BCs, top_x_BCs):
    i = index[0]
    j = index[1]
    if (i == 0):
        return left_y_BCs[j]
    if (i == n_x-1):
        return right_y_BCs[j]
    if (j == 0):
        return bottom_x_BCs[i]
    if (j == n_y-1):
        return top_x_BCs[i]
    raise Exception("tried to get BC on non-boundary node")'''


