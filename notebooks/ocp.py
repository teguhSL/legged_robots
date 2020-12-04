#TODO: modify to accept different Qt and Rt

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
from ocp_sys import *
from scipy.stats import multivariate_normal as mvn
from functools import partial
from scipy.integrate import solve_ivp

from sklearn.neighbors import NearestNeighbors
import quadprog

class ILQR_Standard():
    '''
    ILQR Standard: uses the standard quadratic cost function Q, R, and Qf
    This class is kept only for educational purpose, as it is simpler than the one 
    using cost model
    '''
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def set_ref(self, x_refs):
        self.x_refs = x_refs.copy()
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))
        self.x_refs = np.zeros((self.T+1, self.Dx))

    def set_cost(self, Q, R, Qf = None):
        if Q.ndim == 2:
            self.Q = np.array([Q]*(self.T+1))
            self.R = np.array([R]*(self.T+1)) #note: the last R is only created for convenience, u_T does not affect anything and will be zero
            if Qf is not None:
                self.Q[-1] = Qf
        elif Q.ndim == 3:
            self.Q = Q
            self.R = R
        else:
            print('Number of dimensions must be either 2 or 3')
            #raise()    
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.Lx[i] = self.Q[i].dot(self.xs[i]- self.x_refs[i])
            self.Lxx[i] = self.Q[i]
            self.Luu[i] = self.R[i]
            self.Lu[i] = self.R[i].dot(self.us[i])
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        running_cost_state = 0
        running_cost_control = 0
        cost = 0
        #for i in range(self.T):
        #    cost += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i]) + us[i].T.dot(self.R[i]).dot(us[i])
        #cost += (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        for i in range(self.T):
            running_cost_state += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i])
            running_cost_control += us[i].T.dot(self.R[i]).dot(us[i])
        terminal_cost_state = (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        self.cost = running_cost_state + running_cost_control + terminal_cost_state
        self.running_cost_state = running_cost_state
        self.running_cost_control = running_cost_control
        self.terminal_cost_state = terminal_cost_state
        return self.cost
    
    def calc_dcost(dxs, dus):
        #need to call 'compute_du_LS' first
        return 0.5*dxs.T.dot(self.Qs).dot(dxs) + 0.5*dus.T.dot(self.Rs).dot(dus) + self.Lxs.dot(dxs) + self.Lus.dot(dus)
    
    def forward_pass(self, max_iter = 20):
        print('Starting line searches ...')
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.5
        cost = 5*cost0
        del_us = []
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            xs_new = []
            us_new = []
            x = self.x0.copy()
            xs_new += [x]
            for i in range(self.T):
                del_u = alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                u = self.us[i] + del_u
                x = self.sys.step(x,u)
                xs_new += [x]
                us_new += [u]
                del_us += [del_u]
            
            us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        print('Completing line search ... \n')
            
        self.xs, self.us = np.array(xs_new), np.array(us_new)
        self.del_us = np.array(del_us)
    
    def backward_pass(self):
        self.Vx[self.T] = self.Lx[self.T]
        self.Vxx[self.T] = self.Lxx[self.T]
        for i in np.arange(self.T-1, -1,-1):
            self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
            self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
            self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
            self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            Quuinv_i = inv(self.Quu[i])
            self.k[i] = -Quuinv_i.dot(self.Qu[i])
            self.K[i] = -Quuinv_i.dot(self.Qux[i])

            self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv_i).dot(self.Qux[i])
            self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv_i).dot(self.Qux[i])
            #ensure symmetrical Vxx
            self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)
    
    def solve(self, n_iter = 3):
        for i in range(n_iter):
            self.calc_diff()
            self.backward_pass()
            self.forward_pass()
            
    def compute_du_LS(self):
        self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
        self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))
        
        for i in range(self.T+1):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

        self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
        self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

        #### Calculate Sx and Su 
        i = 0
        self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
        for i in range(1, self.T+1):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

        for i in range(1,self.T+1):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

        self.Lxs = self.Lx.flatten()
        self.Lus = self.Lu.flatten()

        #### Calculate X and U 
        self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs) + self.mu*np.eye(self.Rs.shape[0])
        self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
        self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)
        return self.del_xs_ls, self.del_us_ls, self.Sigma_u_inv
    
    def sample_du(self, n = 1, recreate_dist = True, allow_singular = True):
        print('sampling')
        mean_del_us = self.del_us_ls
        self.Sigma_u = inv(self.Sigma_u_inv)
        self.Sigma_x = self.Su.dot(self.Sigma_u).dot(self.Su.T)
        if recreate_dist:
            self.dist = mvn(mean = mean_del_us, cov = self.Sigma_u, allow_singular = allow_singular)
            
        dx0 = np.zeros(self.Dx)
        Sx0 = self.Sx.dot(dx0)
        sample_dxs = []
        sample_dus = self.dist.rvs(n)
        for i in range(n):
            sample_del_us = sample_dus[i].flatten()
            sample_del_xs = Sx0 + self.Su.dot(sample_del_us)
            sample_dxs += [sample_del_xs]
            
        return  np.array(sample_dxs), np.array(sample_dus)
#missing: better line search and regularization


class ILQR():
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def set_ref(self, x_refs):
        self.x_refs = x_refs.copy()
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))
        self.x_refs = np.zeros((self.T+1, self.Dx))

    def set_cost(self, costs):
        self.costs = costs
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.costs[i].calcDiff(self.xs[i], self.us[i])
            self.Lx[i]  = self.costs[i].Lx
            self.Lxx[i] = self.costs[i].Lxx
            self.Lu[i]  = self.costs[i].Lu
            self.Luu[i] = self.costs[i].Luu
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        self.cost = np.sum([self.costs[i].calc(xs[i], us[i]) for i in range(self.T+1)])
        return self.cost
    
    def calc_dcost(dxs, dus):
        #need to call 'compute_du_LS' first
        return 0.5*dxs.T.dot(self.Qs).dot(dxs) + 0.5*dus.T.dot(self.Rs).dot(dus) + self.Lxs.dot(dxs) + self.Lus.dot(dus)
    
    def forward_pass(self, max_iter = 10):
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.8
        cost = 5*cost0
        
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            xs_new = []
            us_new = []
            x = self.x0.copy()
            xs_new += [x]
            for i in range(self.T):
                u = self.us[i] + alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                x = self.sys.step(x,u)
                xs_new += [x]
                us_new += [u]
            
            us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        self.xs, self.us = np.array(xs_new), np.array(us_new)
    
    
    def backward_pass(self):
        self.Vx[self.T] = self.Lx[self.T]
        self.Vxx[self.T] = self.Lxx[self.T]
        for i in np.arange(self.T-1, -1,-1):
            self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
            self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
            self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
            self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
            Quuinv = inv(self.Quu[i])
            self.k[i] = -Quuinv.dot(self.Qu[i])
            self.K[i] = -Quuinv.dot(self.Qux[i])

            self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv).dot(self.Qux[i])
            self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv).dot(self.Qux[i])
            #ensure symmetrical Vxx
            self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)

    
    
    def solve(self, n_iter = 3):
        for i in range(n_iter):
            self.calc_diff()
            self.backward_pass()
            self.forward_pass()
            
    def compute_du_LS(self):
        self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
        self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))
        
        for i in range(self.T+1):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

        self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
        self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

        #### Calculate Sx and Su 
        i = 0
        self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
        for i in range(1, self.T+1):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

        for i in range(1,self.T+1):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

        self.Lxs = self.Lx.flatten()
        self.Lus = self.Lu.flatten()

        #### Calculate X and U 
        self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs)
        self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
        self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)
        return self.del_xs_ls, self.del_us_ls, self.Sigma_u_inv
    

    def sample_du(self, n = 1, recreate_dist = True):
        mean_del_us = self.del_us_ls
        self.Sigma_u = inv(self.Sigma_u_inv)
        self.Sigma_x = self.Su.dot(self.Sigma_u).dot(self.Su.T)
        if recreate_dist:
            self.dist = mvn(mean = mean_del_us, cov = inv(self.Sigma_u_inv))
            
        dx0 = np.zeros(self.Dx)
        Sx0 = self.Sx.dot(dx0)
        sample_dxs = []
        sample_dus = self.dist.rvs(n)
        
        for i in range(n):
            sample_del_us = sample_dus[i].flatten()
            sample_del_xs = Sx0 + self.Su.dot(sample_del_us)
            sample_dxs += [sample_del_xs]
            
        return np.array(sample_dxs), np.array(sample_dus)

def get_ilqr_from_ddp(ddp, ilqr):
    T = ddp.problem.T
    ilqr.set_timestep(T)
    ilqr.xs = np.array(ddp.xs)
    ilqr.us = np.concatenate([ddp.us, np.zeros((1, len(ddp.us[-1])))])

    datas = ddp.problem.runningDatas
    for i in range(T):
        ilqr.Fx[i] = datas[i].Fx
        ilqr.Fu[i] = datas[i].Fu
        ilqr.Lx[i] = datas[i].Lx.flatten()
        ilqr.Lu[i] = datas[i].Lu.flatten()
        ilqr.Lxx[i] = datas[i].Lxx
        ilqr.Luu[i] = datas[i].Luu

    data = ddp.problem.terminalData
    ilqr.Fx[T] = data.Fx
    ilqr.Fu[T] = data.Fu
    ilqr.Lx[T] = data.Lx.flatten()
    ilqr.Lu[T] = data.Lu.flatten()
    ilqr.Lxx[T] = data.Lxx
    ilqr.Luu[T] = data.Luu
    ilqr.backward_pass()
    return ilqr

class Biped():
    def __init__(self, m1 = 7, m2 = 7, m3 = 17,
                l1 = 0.5, l2 = 0.5, l3 = 0.35, g = 9.81, dT = 0.001, control_opt = 'std'):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.g = g
        self.dT = dT
        self.control_parameters = np.array([457.5, 161, 77.05, 5, 10.4*np.pi/180])  
        self.control_opt = control_opt
        self.Dx = 6
        self.Du = 2
       
    def set_control(self, control_opt = 'std', ilqr = None, xs_ref = None, ddqs_ref = None, Kp = 100*np.eye(3), Kd = 100*np.eye(3), Q = 100*np.eye(3), max_control = 60):
        self.control_opt = control_opt
        if control_opt == 'ilqr':
            self.ilqr = ilqr
        elif control_opt == 'qp':
            self.ilqr = ilqr
            self.xs_ref = xs_ref
            self.Kp = Kp
            self.Kd = Kd
            self.Q = Q
            self.max_control = max_control
            self.ddqs_ref = ddqs_ref
            
        #self.nn = NearestNeighbors(n_neighbors=1)
        #self.nn.fit(ilqr.xs)

    def kin_hip(self, q, dq = None):
        if dq is None: dq = np.zeros(3)
        
        x_h = self.l1*sin(q[0])  
        z_h = self.l1*cos(q[0])  
        dx_h = self.l1*cos(q[0])*dq[0]  
        dz_h = -self.l1*sin(q[0])*dq[0]

        return x_h, z_h, dx_h, dz_h
    
    
    def compute_Jacobian_vhip(self, q, dq):
        J = np.zeros((2,6))
        J[0,0] = -self.l1*sin(q[0])*dq[0]
        J[0,3] = self.l1*cos(q[0])
        J[1,0] = -self.l1*cos(q[0])*dq[0]
        J[1,3] = -self.l1*sin(q[0])
        return J

    def kin_swf(self, q, dq = None):
        if dq is None: dq = np.zeros(3)
        
        x_swf = self.l1*sin(q[0]) - self.l2*sin(q[1])  
        z_swf = self.l1*cos(q[0]) - self.l2*cos(q[1])
        dx_swf = self.l1*cos(q[0])*dq[0] - self.l2*cos(q[1])*dq[1]  
        dz_swf = -self.l1*sin(q[0])*dq[0] + self.l2*sin(q[1])*dq[1]
        return x_swf, z_swf, dx_swf, dz_swf
    
    def compute_Jacobian_swf(self, q, dq = None):
        J = np.zeros((2,3))
        J[0,0] = self.l1*cos(q[0])
        J[0,1] = -self.l2*cos(q[1])
        J[1,0] = -self.l1*sin(q[0])
        J[1,1] = self.l2*sin(q[1])
        return J
    
    
    def eval_A_m(self, q_m):
        t2 = q_m[0]-q_m[1]
        t3 = cos(t2)
        t4 = self.l1**2
        t5 = self.m1*t4*0.25
        t6 = q_m[0] - q_m[2]
        t7 = cos(t6)
        t8 = self.l3**2
        m, m3 = self.m1, self.m3
        l1,l2,l3 = self.l1, self.l2, self.l3
        A_m = np.array([t5-l1*l2*m*t3-l1*l2*m3*t3-l1*l3*m3*t7*(1.0/2.0), 
                        t5,l1*l3*m3*t7*(-1.0/2.0),l2**2*m*(1.0/4.0),0.0,0.0,\
                        m3*t8*(-1.0/4.0)-l2*l3*m3*cos(q_m[1]-q_m[2])*(1.0/2.0),\
                        0.0,m3*t8*(-1.0/4.0)]).reshape(3,3).T
        return A_m
    
    def eval_A_p(self, q_p):
        q1_p = q_p[0]
        q2_p = q_p[1]
        q3_p = q_p[2]
        m, m3, l1, l2, l3 = self.m1, self.m3, self.l1, self.l2, self.l3
        
        t2 = l1**2;
        t3 = q1_p-q2_p;
        t4 = cos(t3);
        t5 = l1*l2*m*t4*(1.0/2.0);
        t6 = q1_p-q3_p;
        t7 = cos(t6);
        t8 = l2**2;
        t9 = l3**2;
        A_p = np.array([t5-m*t2*(5.0/4.0)-m3*t2-l1*l3*m3*t7*(1.0/2.0),t5,l1*l3*m3*t7*(-1.0/2.0),t5-m*t8*(1.0/4.0),
                       m*t8*(-1.0/4.0),0.0,m3*t9*(-1.0/4.0)-l1*l3*m3*t7*(1.0/2.0),0.0,m3*t9*(-1.0/4.0)]).reshape(3,3).T
        return A_p
    
    def eval_B(self):
        B = np.array([1.0,0.0,0.0, 1.0,-1.0,-1.0]).reshape(3,2)
        return B
    
    def eval_C(self, q, dq):
        q1, q2, q3 = q[0], q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]

        m1, m2, m3, l1, l2, l3 =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3

        t2 = q1-q2
        t3 = sin(t2)
        t4 = q1-q3
        t5 = sin(t4)
        C = np.array([0.0,dq1*l1*l2*m2*t3*(1.0/2.0),dq1*l1*l3*m3*t5*(-1.0/2.0),
                      dq2*l1*l2*m2*t3*(-1.0/2.0),0.0,0.0,dq3*l1*l3*m3*t5*(1.0/2.0),0.0,0.0]).reshape(3,3).T
        return C
    
    def eval_energy(self, q, dq):
        q1, q2, q3 = q[0], q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]

        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g

        T = (dq1**2*l1**2*m1)/8 + (dq1**2*l1**2*m2)/2 + (dq1**2*l1**2*m3)/2 + (dq2**2*l2**2*m2)/8 + (dq3**2*l3**2*m3)/8 - (dq1*dq2*l1*l2*m2*cos(q1 - q2))/2 + (dq1*dq3*l1*l3*m3*cos(q1 - q3))/2;
        V = g*m2*(l1*cos(q1) - (l2*cos(q2))/2) + g*m3*(l1*cos(q1) + (l3*cos(q3))/2) + (g*l1*m1*cos(q1))/2;
        return T, V
    
    def eval_G(self, q):
        q1, q2, q3 = q[0], q[1], q[2]
        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g
        
        G = np.array([g*l1*sin(q1)*(m1+m2*2.0+m3*2.0)*(-1.0/2.0)
                      ,g*l2*m2*sin(q2)*(1.0/2.0),g*l3*m3*sin(q3)*(-1.0/2.0)])
        return G
    
    def eval_M(self, q):
        q1, q2, q3 = q[0], q[1], q[2]
        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g

        t2 = q1-q2;
        t3 = cos(t2);
        t4 = q1-q3;
        t5 = cos(t4);
        t6 = l1*l3*m3*t5*(1.0/2.0);
        M = np.array([l1**2*(m1*(1.0/4.0)+m2+m3),l1*l2*m2*t3*(-1.0/2.0),
                      t6,l1*l2*m2*t3*(-1.0/2.0),l2**2*m2*(1.0/4.0),0.0,
                      t6,0.0,l3**2*m3*(1.0/4.0)]).reshape(3,3)
        return M
    
    def impact(self, q_m, dq_m):
        q_p = np.array([q_m[1], q_m[0], q_m[2]])
        A_m = self.eval_A_m(q_m);
        A_p = self.eval_A_p(q_p);
        dq_p = np.linalg.pinv(A_p).dot(A_m).dot(dq_m)
        return q_p, dq_p
    
    def visualize(self, q, r0 = None, fig = None, figsize = (10,10)):
        if fig is None:     
            if self.fig is None: self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig
        if r0 is None: r0 = np.zeros(2)

        x0 = r0[0];
        z0 = r0[1];

        l1, l2, l3 = self.l1, self.l2, self.l3 
        q1, q2, q3 = q[0], q[1], q[2]

        x1 = (l1*sin(q1))/2 + x0;
        z1 = (l1*cos(q1))/2 + z0;
        x2 = l1*sin(q1) - (l2*sin(q2))/2 + x0;
        z2 = l1*cos(q1) - (l2*cos(q2))/2 + z0;
        x3 = l1*sin(q1) + (l3*sin(q3))/2 + x0;
        z3 = l1*cos(q1) + (l3*cos(q3))/2 + z0;

        x_h = l1*sin(q1) + x0;
        z_h = l1*cos(q1) + z0;

        x_t = l1*sin(q1) + l3*sin(q3) + x0;
        z_t = l1*cos(q1) + l3*cos(q3) + z0;

        x_swf = l1*sin(q1) - l2*sin(q2) + x0;
        z_swf = l1*cos(q1) - l2*cos(q2) + z0;

        lw = 5;
        # links
        if self.line1 is None:
            self.line1, = plt.plot([x0, x_h], [z0, z_h], linewidth = lw); 
            self.line2, = plt.plot([x_h, x_t], [z_h, z_t], linewidth = lw); 
            self.line3, = plt.plot([x_h, x_swf], [z_h, z_swf], linewidth = lw);
            # plot a line for "ground"
            self.lineg, = plt.plot([-1 + x_h, 1 + x_h], [0, 0], 'b');
            plt.axis('equal')
            plt.xlim([-1 + x_h, 1 + x_h]);
            plt.ylim([-0.8, 1.2]);
            # point masses
            mz = 15;
            self.marker1, = plt.plot(x1, z1, '.', markersize = mz); 
            self.marker2, = plt.plot(x2, z2, '.',  markersize = mz); 
            self.marker3, = plt.plot(x3, z3, '.',  markersize = mz);
        else:
            self.line1.set_xdata([x0, x_h])
            self.line1.set_ydata([z0, z_h])
            self.line2.set_xdata([x_h, x_t])
            self.line2.set_ydata([z_h, z_t])
            self.line3.set_xdata([x_h, x_swf])
            self.line3.set_ydata([z_h, z_swf])
            self.marker1.set_xdata(x1)
            self.marker1.set_ydata(z1)
            self.marker2.set_xdata(x2)
            self.marker2.set_ydata(z2)
            self.marker3.set_xdata(x3)
            self.marker3.set_ydata(z3)          
            self.lineg.set_xdata( [-1 + x_h, 1 + x_h])
            self.lineg.set_ydata( [0, 0])
            plt.xlim([-1 + x_h, 1 + x_h]);
            self.fig.canvas.draw()
            plt.pause(1e-6)
            self.fig.canvas.flush_events()
        
    def animate(self, sln, dt = 0.01, skip = 10):
        tic = time.time();
        self.line1, self.line2, self.line3 = None, None, None
        self.marker1, self.marker2, self.marker3 = None, None, None
        self.fig = plt.figure(figsize=(10,10))

        num_steps = len(sln['T'])
        r0 = np.zeros(2)
        for j in range(num_steps):
            Y = sln['Y'][j]
            N = len(Y)
            for i in range(0, N, skip):
                q = Y[i, :3]
                time.sleep(dt)
                self.visualize(q, r0, fig = self.fig)
            x0, _,_,_ = self.kin_swf(q)
            r0 = r0 + np.array([x0, 0])

        toc = time.time()
        
        real_time_factor = sln['T'][-1][-1] / (toc-tic)
        print('Real time factor:{}'.format(real_time_factor))
        return real_time_factor
    
    def animate_ys(self, ys, skip = 10, dt = 0.001,figsize=(10,10)):
        fig = plt.figure(figsize = figsize)
        self.fig = fig
        self.line1 = None
        z_swfs = []
        for y in ys[::skip]:
            self.visualize(y[:3], fig=fig)
            _,z_swf,_,_ = self.kin_swf(y[:3])
            z_swfs += [z_swf]
            time.sleep(dt)
        z_swfs = np.array(z_swfs)
        return z_swfs
    
    def analyse(self, sln, parameters,  to_plot = True, Umax = 30):
        x_hs, dx_hs, dx_hs_mean, z_hs, dz_hs = [], [], [], [], []
        x_swfs, dx_swfs, z_swfs, dz_swfs = [], [], [], []      
        qs, dqs, us = [], [], []
        t_hs = [] #time index 
        dts, fs, ds = [], [], [] # time interval for one step, frequency and displacements
        effort = 0
        cot = 0

        num_step = len(sln['Y'])
        for i in range(num_step):
            Y, T = sln['Y'][i], sln['T'][i]
            t0 = T[0]
            t_hs += [T]
            dt = T[-1] - T[0]
            dts += [dt]
            fs += [1.0/dt]

            for j in range(len(Y)):
                y, t = Y[j], T[j]
                
                x_h, z_h, dx_h, dz_h =  self.kin_hip(y[:3], y[3:])
                x_hs += [x_h]
                dx_hs += [dx_h]
                z_hs += [z_h]
                dz_hs += [dz_h]

                x_swf, z_swf, dx_swf, dz_swf =  self.kin_swf(y[:3], y[3:])
                x_swfs += [x_swf]
                dx_swfs += [dx_swf]
                z_swfs += [z_swf]
                dz_swfs += [dz_swf]
                
                qs += [y[:3]]
                dqs += [y[3:]]
                if self.control_opt == 'std':
                    u = self.control(t, y[:3], y[3:],parameters);
                elif self.control_opt == 'ilqr':
                    u = self.control_ilqr(t, y, t0);
                elif self.control_opt == 'qp':
                    u = self.control_qp(t, y, t0);
                    
                    
                us += [u]

            x_h0, _, _, _ =  self.kin_hip(Y[0][:3], Y[0][3:])
            x_hT, _, _, _ =  self.kin_hip(Y[-1][:3], Y[-1][3:])
            d = x_hT - x_h0
            ds += [d]
            dx_hs_mean += [d/dt]

        results = dict();
        results['x_h'], results['dx_h'], results['z_h'], results['dz_h'], results['dx_hs_mean']  = np.array(x_hs), np.array(dx_hs), np.array(z_hs), np.array(dz_hs), np.array(dx_hs_mean)
        results['x_swf'], results['dx_swf'], results['z_swf'], results['dz_swf']  = np.array(x_swfs), np.array(dx_swfs), np.array(z_swfs), np.array(dz_swfs)
        results['t_h'] = np.concatenate(t_hs)
        
        results['qs'], results['dqs'], results['us']  = np.array(qs), np.array(dqs), np.array(us)
        results['dts'], results['fs'], results['ds'] = np.array(dts), np.array(fs), np.array(ds)
        T = results['t_h'][-1]
        results['total_mean_velocity'] = np.sum(results['ds'])/T
        results['last_mean_velocity'] = results['ds'][-1]/results['dts'][-1]
        results['effort'] = np.sum(results['us'][:,0]**2 + results['us'][:,1]**2)/(2*T*Umax)
        results['cot'] = results['effort']/np.sum(results['ds'])
        
        if to_plot:
            fig, axs = plt.subplots(4, 5, figsize=(20,25))

            axs[0,0].plot(results['t_h'], (results['x_h']))
            axs[0,0].set_title('xh')
            axs[0,1].plot(results['t_h'], (results['dx_h']))
            axs[0,1].set_title('dxh')
            axs[0,2].plot(results['t_h'], (results['z_h']))
            axs[0,2].set_title('zh')
            axs[0,3].plot(results['t_h'], (results['dz_h']))
            axs[0,3].set_title('dzh')
            axs[0,4].plot(results['dx_hs_mean'])
            axs[0,4].set_title('dxh_mean')
            
            axs[1,0].plot(results['t_h'], (results['x_swf']))
            axs[1,0].set_title('x_swf')
            axs[1,1].plot(results['t_h'], (results['dx_swf']))
            axs[1,1].set_title('dx_swf')
            axs[1,2].plot(results['t_h'], (results['z_swf']))
            axs[1,2].set_title('z_swf')
            axs[1,3].plot(results['t_h'], (results['dz_swf']))
            axs[1,3].set_title('dz_swf')
            
            for j in range(3):
                axs[2,j].plot(results['qs'][:,j], results['dqs'][:,j])
                axs[2,j].set_title('phase plot '+str(j))
            for j in range(2):
                axs[2,3+j].plot(results['t_h'], (results['us'][:,j]))
                axs[2,3+j].set_title('us'+str(j))
           
            axs[3,0].plot((results['dts'][1:]), '-o')
            axs[3,0].set_title('dT')
            axs[3,1].plot((results['fs'][1:]), '-o')
            axs[3,1].set_title('fs')
            axs[3,2].plot((results['ds'][0:]), '-o')
            axs[3,2].set_title('ds')
            
        print('COT: {0:.2f}, Dist: {1:.3f}, dT: {2:.3f}'.format(results['cot'], results['ds'][-1], results['dts'][-1]))
        return results
    
    
    def control(self, t, q, dq, parameters):
        kp1 = parameters[0];
        kp2 = parameters[1];
        kd1 = parameters[2];
        kd2 = parameters[3];
        alpha = parameters[4];

        y1 = q[2]-alpha;
        y1d = dq[2];
        y2 = -q[1]-q[0];
        y2d = -dq[1]-dq[0];

        u1 = kp1*y1 + kd1*y1d;

        u2 = kp2*y2 + kd2*y2d;

        u1 = max(min(u1, 30), -30) 
        u2 = max(min(u2, 30), -30) 
        u = np.array([u1, u2])
        return u
    
    def compute_control_standard(self, x0, T, params):
        #compute xs and us from standard controller
        x, us, z_swfs = x0.copy(), [], []
        x_h0, _,_,_ = self.kin_hip(x[:3])
        for k in range(T):
            u = self.control(0, x[:3], x[3:], params)
            x = self.step(x, u)
            _,z_swf,_,_ = self.kin_swf(x[:3])
            z_swfs += [z_swf]
            us += [u]
        x_hT , _,dx_hT,dz_hT = self.kin_hip(x[:3])
        dist = x_hT-x_h0
        
        us, z_swfs = np.array(us), np.array(z_swfs)
        self.set_init_state(x0)
        xs = self.rollout(us)
        #add a dummy zero control at the end
        us = np.concatenate([us, np.zeros((1, self.Du))], axis=0)
        return xs, us, z_swfs, dist
    
    def compute_inverse_dynamics(self, ys, ddq_ds):
        N = len(ys)
        taus = []
        for i in range(N):
            y, ddq_d = ys[i], ddq_ds[i]
            q, dq = y[:3], y[3:]
            M = self.eval_M(q)
            C = self.eval_C(q, dq)
            G = self.eval_G(q)
            B = self.eval_B()

            tau = np.linalg.pinv(B).dot(M.dot(ddq_d) + C.dot(dq) + G)
            taus += [tau]
        return taus
    
    

        
    
    def step(self, y, u):
        #step using Euler integration
        q = y[:3]
        dq = y[3:]
        M = self.eval_M(q)
        C = self.eval_C(q, dq)
        G = self.eval_G(q)
        B = self.eval_B()
        dy = np.zeros(6)
        dy[:3] = y[3:]
        dy[3:] = np.linalg.solve(M, -C.dot(dq) - G + B.dot(u))
        ynext = y + dy*self.dT
        return ynext
    
    def set_init_state(self,x0):
        self.x0 = x0
    
    def rollout(self,us):
        #rollout using Euler integration
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def compute_matrices(self, y, u, inc = 1e-3):
        Dx = len(y)
        Du = len(u)
        A = np.zeros((Dx, Dx))
        B = np.zeros((Dx, Du))
        
        #compute A
        y0, u0 = y.copy(), u.copy()
        for i in range(Dx):
            y0p = y0.copy()
            y0p[i] += inc
            dyp = self.step(y0p,u0)

            y0m = y0.copy()
            y0m[i] -= inc
            dym = self.step(y0m,u0)
            
            diff = (dyp - dym)/(2*inc)
            
            A[:,i] = diff
            
        #compute B
        y0, u0 = y.copy(), u.copy()
        for i in range(Du):
            u0p = u0.copy()
            u0p[i] += inc
            dup = self.step(y0,u0p)

            u0m = u0.copy()
            u0m[i] -= inc
            dum = self.step(y0,u0m)
            
            diff = (dup - dum)/(2*inc)
            
            B[:,i] = diff
            
        return A, B
            
    def solve_eqns(self, q0, dq0, num_steps, parameters, retrain = False, retrain_iter = 5):
        
        #options = odeset('RelTol',1e-5, 'Events', @event_func);
        h = self.dT; # time step
        tmax = 2; # max time that we allow for a single step
        tspan = np.arange(0, tmax, h)
        y0 = np.concatenate([q0, dq0])
        t0 = 0

        sln = {'T' : [], 'Y' : [], 'TE' : [], 'YE' : []}
        
        for i in range(num_steps):
            tic = time.time()
            eqns_std = partial(self.eqns, t0 = t0, parameters = parameters )
            if self.control_opt == 'ilqr' and retrain == True:
                # retrain ilqr using the initial guess from standard controller
                self.ilqr.set_init_state(y0.copy())
                self.set_init_state(y0.copy())
                #us = np.zeros((self.ilqr.T+1, self.Du))
                x = y0.copy()
                us = []
                for k in range(self.ilqr.T+1):
                    u = self.control(0, x[:3], x[3:], params)
                    x = self.step(x, u)
                    us += [u]
                us = np.array(us)
                xs = self.rollout(us)
                us = np.concatenate([us, np.zeros((1, self.Du))], axis=0)
                #xs[0] = y0.copy()
                self.ilqr.set_state(xs, us)
                self.ilqr.solve(retrain_iter)
                clear_output()
            tspan = np.arange(t0, t0+tmax-1e-4, h)
            sol = solve_ivp(eqns_std, (t0, t0+tmax),  y0, t_eval = np.arange(t0, t0+tmax-1e-4, h), events = event_func, rtol = 1e-5)            
            toc = time.time()
            sln['T'] += [sol.t]
            sln['Y'] += [sol.y.T]
            sln['TE'] += [sol.t_events]

            if np.abs(sol.t[-1]- tmax) < 1e-4:
                break

            if len(sol.t_events) == 0:
                break

            # Impact map
            q_m = sol.y.T[-1,:3]
            dq_m = sol.y.T[-1,3:]
            q_p, dq_p = self.impact(q_m, dq_m)

            y0 = np.concatenate([q_p, dq_p])
            t0 = sol.t[-1]
            tspan = np.arange(t0, t0+tmax-1e-4, h)
        return sln
    

    
    def control_ilqr(self, t, y, t0):
        t_step = int((t- t0)/self.dT)
        #print(t_step)
        if t_step >= len(self.ilqr.K):
            t_step = len(self.ilqr.K)-1
        #calculate t_step based on the nearest neighbor
        #dist = np.linalg.norm(y - ilqr.xs, axis=1)
        #t_step = np.argmin(dist)
#         dist, idx = self.nn.kneighbors(y[None,:])
#         t_step = idx[0][0]
#         #print(t_step)
#         if t_step >= len(self.ilqr.K):
#             t_step = len(self.ilqr.K)-1
        
        u = self.ilqr.us[t_step] + self.ilqr.K[t_step].dot(y-self.ilqr.xs[t_step])
        return u
    
    def control_qp(self, t, y, t0):
        t_step = int((t- t0)/self.dT)
        if t_step >= len(self.xs_ref-1):
            t_step = len(self.xs_ref)-1        
        y_ref = self.xs_ref[t_step]
        ddq_d = self.ddqs_ref[t_step]
        u = self.compute_qp(y, y_ref, ddq_d)
        return u
    
    def compute_qp(self, y, yd, ddq_d):
        q, dq, qd, dqd = y[:3], y[3:], yd[:3], yd[3:]
        qdd = ddq_d -self.Kp.dot(q-qd) - self.Kd.dot(dq-dqd)

        xd = np.concatenate([qdd, np.zeros(2)])
        Qt = np.eye(5)*1e-10
        Qt[:3,:3] = self.Q

        a = xd.T.dot(Qt)

        M = self.eval_M(q)
        C = self.eval_C(q,dq)
        G = self.eval_G(q)
        B = self.eval_B()
        
        #use inverse dynamics instead
        #tau = np.linalg.pinv(B).dot(M.dot(qdd) + C.dot(dq) + G)
        #pdb.set_trace()
        
        Ct = np.hstack([M, -B]).T
        Et = np.zeros((5,4))
        Et[3,0] = 1
        Et[3,1] = -1
        Et[3,2] = 1
        Et[3,3] = -1
        Ct = np.hstack([Ct, Et])

        dt = -self.max_control*np.ones(4)
        bt = -C.dot(dq) - G
        bt = np.concatenate([bt, dt])
        #print(Qt)
        x = quadprog.solve_qp(Qt, a, Ct, bt, meq = 3)
        #print('x qp:', x)
        return x[0][-2:]
        
        return tau
    
    def eqns(self, t, y, t0,  parameters):    
        q = y[:3]
        dq = y[3:]
        
        M = self.eval_M(q)
        C = self.eval_C(q, dq)
        G = self.eval_G(q)
        B = self.eval_B()
        if self.control_opt == 'std':
            u = self.control(t, q, dq, parameters)
        elif self.control_opt == 'ilqr':
            u = self.control_ilqr(t, y, t0)
        elif self.control_opt == 'qp':
            u = self.control_qp(t, y, t0)
        n = 6
        dy = np.zeros(n)
        dy[:3] = y[3:]
        #dy[3:] = np.linalg.inv(M).dot(-C.dot(dq) - G + B.dot(u))
        dy[3:] = np.linalg.solve(M, -C.dot(dq) - G + B.dot(u))
        return dy
    
    def compute_ddq(self, y, u):
        #step using Euler integration
        q = y[:3]
        dq = y[3:]
        M = self.eval_M(q)
        C = self.eval_C(q, dq)
        G = self.eval_G(q)
        B = self.eval_B()
        dy = np.zeros(6)
        dy[:3] = y[3:]
        dy[3:] = np.linalg.solve(M, -C.dot(dq) - G + B.dot(u))
        return dy


    
def kin_swf(q, dq = None):
    l1, l2 = 0.5, 0.5
    if dq is None: dq = np.zeros(3)
    
    x_swf = l1*sin(q[0]) - l2*sin(q[1])  
    z_swf = l1*cos(q[0]) - l2*cos(q[1])
    dx_swf = l1*cos(q[0])*dq[0] - l2*cos(q[1])*dq[1]  
    dz_swf = -l1*sin(q[0])*dq[0] + l2*sin(q[1])*dq[1]
    return x_swf, z_swf, dx_swf, dz_swf

def event_func(t, y):
    q = y[:3]
    dq = y[3:]
    _, z_swf, _,_ = kin_swf(q, dq)
    value = z_swf + 0.01 #0.01* cos(q[0]) + 0.0001
    return value
event_func.terminal = 1
event_func.direction = -1

