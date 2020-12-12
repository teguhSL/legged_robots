import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import casadi
from casadi import mtimes, MX, sin, cos, vertcat, horzcat, sum1, cross, Function, jacobian, solve
from utils import *
from functools import partial
from scipy.integrate import solve_ivp
import pinocchio
import crocoddyl
import quadprog

class LinearSystem():
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.Dx = A.shape[0]
        self.Du = B.shape[1]
        
    def reset_AB(self, A,B):
        self.A = A
        self.B = B
        
    def set_init_state(self,x0):
        self.x0 = x0
    
    def step(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

class Unicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 3
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(3)
        A[0,2] = -u[0]*np.sin(x[2])*self.dt
        A[1,2] = u[0]*np.cos(x[2])*self.dt
        
        B = np.zeros((3,2))
        B[0,0] = np.cos(x[2])*self.dt
        B[1,0] = np.sin(x[2])*self.dt
        B[2,1] = 1*self.dt
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        #A,B = self.compute_matrices(x,u)
        x_next = np.zeros(3)
        
        x_next[0] = x[0] + u[0]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + u[0]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + u[1]*self.dt
        #pdb.set_trace()
        return x_next
        #return A.dot(x) + B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class SecondUnicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 5
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[0,2] = -x[3]*np.sin(x[2])*self.dt
        A[1,2] = x[3]*np.cos(x[2])*self.dt
        
        A[0,3] = np.cos(x[2])*self.dt
        A[1,3] = np.sin(x[2])*self.dt
        A[2,4] = 1*self.dt
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = self.dt
        B[4,1] = self.dt
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + x[3]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + x[4]*self.dt
        x_next[3] = x[3] + u[0]*self.dt
        x_next[4] = x[4] + u[1]*self.dt
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class Pendulum():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 2
        self.Du = 1
        self.b = 1
        self.m = 1
        self.l = 1
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,1] = self.dt
        A[1,0] = 0.5*9.8*self.dt*np.cos(x[0])/self.l
        A[1,1] = 1 - self.dt*self.b/(self.m*self.l**2)
        
        B[1,0] = self.dt/(self.m*self.l**2)
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        x_next[0] = x[0] + x[1]*self.dt
        x_next[1] = (1-self.dt*self.b/(self.m*self.l**2))*x[1] + 0.5*9.8*self.dt*np.sin(x[0])/self.l + self.dt*u/(self.m*self.l**2) 
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def plot(self, x, color='k'):
        px = np.array([0, -self.l*np.sin(x[0])])
        py = np.array([0, self.l*np.cos(x[0])])
        line = plt.plot(px, py, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-2*self.l, 2*self.l]
        plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line

    def plot_traj(self, xs, dt = 0.1, filename = None):
        for i,x in enumerate(xs):
            clear_output(wait=True)
            self.plot(x)
            if filename is not None:
                plt.savefig('temp/fig'+str(i)+'.png')
            plt.show()
            time.sleep(dt)
    
    
class Bicopter():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 6
        self.Du = 2
        
        self.m = 2.5
        self.l = 1
        self.I = 1.2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[:3, 3:] = np.eye(3)*self.dt
        A[3,2] = -self.dt*(u[0]+u[1])*np.cos(x[2])/self.m
        A[4,2] = -self.dt*(u[0]+u[1])*np.sin(x[2])/self.m
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = -self.dt*np.sin(x[2])/self.m
        B[3,1] = B[3,0]
        
        B[4,0] = self.dt*np.cos(x[2])/self.m
        B[4,1] = B[4,0]
        
        B[5,0] = self.dt*self.l*0.5/self.I
        B[5,1] = -B[5,0]
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*self.dt
        x_next[1] = x[1] + x[4]*self.dt
        x_next[2] = x[2] + x[5]*self.dt
        
        x_next[3] = x[3] - (u[0]+u[1])*np.sin(x[2])*self.dt/self.m
        x_next[4] = x[4] + (u[0]+u[1])*np.cos(x[2])*self.dt/self.m - 9.8*self.dt
        x_next[5] = x[5] + (u[0]-u[1])*self.dt*self.l*0.5/self.I
        
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
        
    def plot(self, x, color = 'k'):
        pxs = np.array([x[0] + 0.5*self.l*np.cos(x[2]), x[0] - 0.5*self.l*np.cos(x[2])])
        pys = np.array([x[1] + 0.5*self.l*np.sin(x[2]), x[1] - 0.5*self.l*np.sin(x[2])])
        line = plt.plot(pxs, pys, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        return line

    def vis_traj(self, xs, dt = 0.1, axes_lim = [-5,5,-5,5]):
        T = len(xs)
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.axes().set_aspect('equal')
            plt.axis(axes_lim)
            plt.show()
            time.sleep(dt)
            

class Quadcopter():
    def __init__(self, dt = 0.01, I = np.diag(np.array([2,2,4])), kd = 1, 
                 k = 1, L = 0.3, b = 1, m=1, g=9.81, Dx=12, Du=4):
        self.I = I #inertia
        self.kd = kd #friction
        self.k = k #motor constant
        self.L = L# distance between center and motor
        self.b = b # drag coefficient
        self.m = m # mass
        self.g = g
        self.Dx = Dx
        self.Du = Du
        self.dt = dt
        
    def thrust(self, inputs):
        T = np.array([0,0, self.k*np.sum(inputs)])
        return T

    def torques(self, inputs):
        tau = np.array([self.L*self.k*(inputs[0]-inputs[2]), self.L*self.k*(inputs[1]-inputs[3]), \
                        self.b*(inputs[0]-inputs[1] + inputs[2] - inputs[3])])
        return tau

    def acceleration(self, inputs, angles, xdot):
        gravity = np.array([0,0,-self.g])
        R = self.Rotation(angles)
        T = R.dot(self.thrust(inputs))
        Fd = -self.kd*xdot
        a = gravity + T/self.m + Fd
        return a

    def angular_acceleration(self, inputs, omega):
        tau = self.torques(inputs)
        omegadot = np.linalg.inv(self.I).dot(tau - np.cross(omega, self.I.dot(omega)))
        return omegadot

    def thetadot2omega(self, thetadot, theta):
        R = np.array([[1, 0, -np.sin(theta[1])], \
                     [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])], \
                     [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]])
        return R.dot(thetadot)

    def omega2thetadot(self, omega, theta):
        R = np.array([[1, 0, -np.sin(theta[1])], \
                     [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])], \
                     [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]])
        return np.linalg.inv(R).dot(omega)

    def Rotation(self, theta):
        c0,s0 = np.cos(theta[0]), np.sin(theta[0])
        c1,s1 = np.cos(theta[1]), np.sin(theta[1])
        c2,s2 = np.cos(theta[2]), np.sin(theta[2])

        R = np.array([[c0*c2 - c1*s0*s2, -c2*s0 - c0*c1*s2, s1 * s2], 
                     [c1*c2*s0 + c0*s2, c0*c1*c2-s0*s2, -c2*s1], 
                     [s0*s1, c0*s1, c1]])
        return R
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u, inc = 0.001):
        Dx, Du = len(x), len(u)
        A = np.zeros((Dx, Dx))
        B = np.zeros((Dx, Du))
        
        xnext = self.step(x, u)
        for i in range(Dx):
            xp, xm = x.copy(), x.copy()
            xp[i] += inc
            xnextp = self.step(xp, u)
            xm[i] -= inc
            xnextm = self.step(xm, u)
            diff = (xnextp - xnextm)/(2*inc)
            A[:,i] = diff
            
        for i in range(Du):
            up, um = u.copy(), u.copy()
            up[i] += inc
            xnextp = self.step(x, up)
            um[i] -= inc
            xnextm = self.step(x, um)
            diff = (xnextp - xnextm)/(2*inc)
            B[:,i] = diff
        
        return A,B
        
    def step(self, x, u, u_offset = None):
        if u_offset is None:
            u_mag = np.sqrt(9.81/4)
            u_offset = np.array([u_mag]*self.Du)**2 
        u_act = u_offset + u**2
        p, pdot, theta, thetadot = x[:3], x[3:6], x[6:9], x[9:]

        #step
        omega = self.thetadot2omega(thetadot, theta)
    
        a = self.acceleration(u_act, theta, pdot)
        omegadot = self.angular_acceleration(u_act, omega)
        omega = omega + self.dt*omegadot
        thetadot= self.omega2thetadot(omega, theta)
        theta = theta + self.dt*thetadot
        pdot = pdot + self.dt*a
        p = p + self.dt*pdot
        
        x_next = np.concatenate([p, pdot, theta, thetadot])
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def vis_quad(self, quadId, xs, dt = 0.05):
        for i,x in enumerate(xs):
            ori = euler2quat(xs[i,6:9], 'rzyz')
            p.resetBasePositionAndOrientation(quadId, xs[i,:3], ori)
            time.sleep(dt)
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, 
                                         cameraPitch=-20, cameraTargetPosition=xs[i,:3])
            
class ActionModelRobot(crocoddyl.ActionModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ActionModelAbstract.__init__(self, state, nu)
        
    def init_robot_sys(self,robot_sys, nr = 1):
        self.robot_sys = robot_sys
        self.Du = robot_sys.Du
        self.Dx = robot_sys.Dx
        self.Dr = nr
        
    def set_cost(self, cost_model):
        self.cost_model = cost_model
        
    def calc(self, data, x, u):
        #calculate the cost
        data.cost = self.cost_model.calc(x,u)
        
        #calculate the next state
        data.xnext = self.robot_sys.step(x,u)
        
    def calcDiff(self, data, x, u, recalc = False):
        if recalc:
            self.calc(data, x, u)

        #compute cost derivatives
        self.cost_model.calcDiff(x, u)
        data.Lx = self.cost_model.Lx.copy()
        data.Lxx = self.cost_model.Lxx.copy()
        data.Lu = self.cost_model.Lu.copy()
        data.Luu = self.cost_model.Luu.copy()
        
        #compute dynamic derivatives 
        A, B = self.robot_sys.compute_matrices(x,u)
        data.Fx = A.copy()
        data.Fu = B.copy()
        
    def createData(self):
        data = ActionDataRobot(self)
        return data

class ActionDataRobot(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self,model)
        
        

class Biped_Casadi():
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
        
        self.x = MX.sym('x', self.Dx)
        self.u = MX.sym('u', self.Du)
        
        self.use_noise = False
        self.noise_idx = 0
        self.noise = None
       
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
            
    def set_noise(self, noise):
        self.q_noise = noise[:,:3]
        self.dq_noise = noise[:,3:]
        self.noise_idx = 0
        self.use_noise = True
            
    def compute_Jacobian_swf(self, q, dq = None):
        J = np.zeros((2,3))
        J[0,0] = self.l1*cos(q[0])
        J[0,1] = -self.l2*cos(q[1])
        J[1,0] = -self.l1*sin(q[0])
        J[1,1] = self.l2*sin(q[1])
        return J
    def compute_Jacobian_vhip(self, q, dq):
        J = np.zeros((2,6))
        J[0,0] = -self.l1*sin(q[0])*dq[0]
        J[0,3] = self.l1*cos(q[0])
        J[1,0] = -self.l1*cos(q[0])*dq[0]
        J[1,3] = -self.l1*sin(q[0])
        return J

    def eval_energy(self, q, dq):
        q1, q2, q3 = q[0], q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]

        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g

        T = (dq1**2*l1**2*m1)/8 + (dq1**2*l1**2*m2)/2 + (dq1**2*l1**2*m3)/2 + (dq2**2*l2**2*m2)/8 + (dq3**2*l3**2*m3)/8 - (dq1*dq2*l1*l2*m2*cos(q1 - q2))/2 + (dq1*dq3*l1*l3*m3*cos(q1 - q3))/2;
        V = g*m2*(l1*cos(q1) - (l2*cos(q2))/2) + g*m3*(l1*cos(q1) + (l3*cos(q3))/2) + (g*l1*m1*cos(q1))/2;
        return T, V

    def kin_hip(self, q, dq = None):
        if dq is None: dq = np.zeros(3)
        
        x_h = self.l1*sin(q[0])  
        z_h = self.l1*cos(q[0])  
        dx_h = self.l1*cos(q[0])*dq[0]  
        dz_h = -self.l1*sin(q[0])*dq[0]

        return x_h, z_h, dx_h, dz_h
    
    def kin_swf(self, q, dq = None):
        if dq is None: dq = np.zeros(3)
        
        x_swf = self.l1*sin(q[0]) - self.l2*sin(q[1])  
        z_swf = self.l1*cos(q[0]) - self.l2*cos(q[1])
        dx_swf = self.l1*cos(q[0])*dq[0] - self.l2*cos(q[1])*dq[1]  
        dz_swf = -self.l1*sin(q[0])*dq[0] + self.l2*sin(q[1])*dq[1]
        return x_swf, z_swf, dx_swf, dz_swf
    
    
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
        A_m = vertcat(t5-l1*l2*m*t3-l1*l2*m3*t3-l1*l3*m3*t7*(1.0/2.0), 
                        t5,l1*l3*m3*t7*(-1.0/2.0),l2**2*m*(1.0/4.0),0.0,0.0,\
                        m3*t8*(-1.0/4.0)-l2*l3*m3*cos(q_m[1]-q_m[2])*(1.0/2.0),\
                        0.0,m3*t8*(-1.0/4.0)).reshape((3,3))
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
        A_p = vertcat(t5-m*t2*(5.0/4.0)-m3*t2-l1*l3*m3*t7*(1.0/2.0),t5,l1*l3*m3*t7*(-1.0/2.0),t5-m*t8*(1.0/4.0),
                       m*t8*(-1.0/4.0),0.0,m3*t9*(-1.0/4.0)-l1*l3*m3*t7*(1.0/2.0),0.0,m3*t9*(-1.0/4.0)).reshape((3,3))
        return A_p
    
    def eval_B(self, mode = 'casadi'):
        if mode == 'casadi':
            B = vertcat(1.0,0.0,0.0, 1.0,-1.0,-1.0).reshape((2,3)).T
        elif mode == 'numpy':
            B = np.array([1.0,0.0,0.0, 1.0,-1.0,-1.0]).reshape(3,2)
        return B
    
    def eval_C(self, q, dq, mode = 'casadi'):
        q1, q2, q3 = q[0], q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]

        m1, m2, m3, l1, l2, l3 =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3

        t2 = q1-q2
        t3 = sin(t2)
        t4 = q1-q3
        t5 = sin(t4)
        if mode == 'casadi':
            C = vertcat(0.0,dq1*l1*l2*m2*t3*(1.0/2.0),dq1*l1*l3*m3*t5*(-1.0/2.0),
                      dq2*l1*l2*m2*t3*(-1.0/2.0),0.0,0.0,dq3*l1*l3*m3*t5*(1.0/2.0),0.0,0.0).reshape((3,3))
        elif mode == 'numpy':
            C = np.array([0.0,dq1*l1*l2*m2*t3*(1.0/2.0),dq1*l1*l3*m3*t5*(-1.0/2.0),
                      dq2*l1*l2*m2*t3*(-1.0/2.0),0.0,0.0,dq3*l1*l3*m3*t5*(1.0/2.0),0.0,0.0]).reshape(3,3).T
        return C
        
    
    def eval_G(self, q, mode = 'casadi'):
        q1, q2, q3 = q[0], q[1], q[2]
        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g
        
        if mode == 'casadi':
            G = vertcat(g*l1*sin(q1)*(m1+m2*2.0+m3*2.0)*(-1.0/2.0)
                          ,g*l2*m2*sin(q2)*(1.0/2.0),g*l3*m3*sin(q3)*(-1.0/2.0))
        elif mode == 'numpy':
            G = np.array([g*l1*sin(q1)*(m1+m2*2.0+m3*2.0)*(-1.0/2.0)
                          ,g*l2*m2*sin(q2)*(1.0/2.0),g*l3*m3*sin(q3)*(-1.0/2.0)])
            
        return G
    
    def eval_M(self, q, mode = 'casadi'):
        q1, q2, q3 = q[0], q[1], q[2]
        m1, m2, m3, l1, l2, l3, g =  self.m1, self.m2,  self.m3, self.l1, self.l2, self.l3, self.g

        t2 = q1-q2;
        t3 = cos(t2);
        t4 = q1-q3;
        t5 = cos(t4);
        t6 = l1*l3*m3*t5*(1.0/2.0);
        if mode == 'casadi':
            M = vertcat(l1**2*(m1*(1.0/4.0)+m2+m3),l1*l2*m2*t3*(-1.0/2.0),
                          t6,l1*l2*m2*t3*(-1.0/2.0),l2**2*m2*(1.0/4.0),0.0,
                          t6,0.0,l3**2*m3*(1.0/4.0)).reshape((3,3)).T
        elif mode == 'numpy':
            M = np.array([l1**2*(m1*(1.0/4.0)+m2+m3),l1*l2*m2*t3*(-1.0/2.0),
                          t6,l1*l2*m2*t3*(-1.0/2.0),l2**2*m2*(1.0/4.0),0.0,
                          t6,0.0,l3**2*m3*(1.0/4.0)]).reshape(3,3)
            
        return M
    
    def impact(self, q_m, dq_m):
        q_p = np.array([q_m[1], q_m[0], q_m[2]])
        A_m = self.eval_A_m(q_m)
        A_p = self.eval_A_p(q_p)
        dq_p = np.linalg.pinv(A_p).dot(A_m).dot(dq_m)
        return q_p, dq_p
    
    def step_cas(self, y, u):
        #step using Euler integration
        q = y[:3]
        dq = y[3:]
        M = self.eval_M(q)
        C = self.eval_C(q, dq)
        G = self.eval_G(q)
        B = self.eval_B()
        dy = vertcat( y[3:], solve(M, -mtimes(C,dq) - G + mtimes(B,u)))
        ynext = y + dy*self.dT
        return ynext

    def step(self, y, u):
        #step using Euler integration
        q = y[:3]
        dq = y[3:]
        mode = 'numpy'
        M = self.eval_M(q, mode)
        C = self.eval_C(q, dq, mode)
        G = self.eval_G(q, mode)
        B = self.eval_B(mode)
        dy = np.concatenate([ y[3:], np.linalg.solve(M, -C.dot(dq) - G + B.dot(u))])
        ynext = y + dy*self.dT
        return ynext
    
    def def_compute_matrices(self):
        self.ynext = self.step_cas(self.x, self.u)
        self.A = jacobian(self.ynext, self.x)
        self.B = jacobian(self.ynext, self.u)
        self.A_val = Function('A', [self.x,self.u], [self.A])
        self.B_val = Function('B', [self.x,self.u], [self.B])
        
    def compute_matrices(self,x,u):
        A = np.array(self.A_val(x,u))
        B = np.array(self.B_val(x,u))
        return A, B
    
            
    
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
    
    def analyse(self, sln, parameters=None,  to_plot = True, Umax = 30, to_save = False, filename = ''):
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
            fig, axs = plt.subplots(5, 6, figsize=(25,30))

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

            for j in range(3):
                axs[4,j].plot(results['qs'][:,j])
                axs[4,j].set_title('angle plot '+str(j))
            
            for j in range(3, 6):
                axs[4,j].plot(results['dqs'][:,j-3])
                axs[4,j].set_title('angle vel plot '+str(j-3))

            
        print('COT: {0:.2f}, Dist: {1:.3f}, dT: {2:.3f}'.format(results['cot'], results['ds'][-1], results['dts'][-1]))
        if to_save:
            plt.savefig(filename, dpi = 500)
        return results
    
    
    def control(self, t, q, dq, parameters):
        #add noises
        if self.use_noise is True:
            q = q + self.q_noise[self.noise_idx]
            dq = dq + self.dq_noise[self.noise_idx]
            self.noise_idx += 1
            if self.noise_idx == len(self.q_noise):
                self.noise_idx = 0
        
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
        mode = 'numpy'
        B = self.eval_B(mode)
        for i in range(N):
            y, ddq_d = ys[i], ddq_ds[i]
            q, dq = y[:3], y[3:]
            M = self.eval_M(q, mode)
            C = self.eval_C(q, dq, mode)
            G = self.eval_G(q, mode)

            tau = np.linalg.pinv(B).dot(M.dot(ddq_d) + C.dot(dq) + G)
            taus += [tau]
        return taus
    
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

    def solve_eqns(self, q0, dq0, num_steps, parameters, retrain = False, retrain_iter = 5):
        self.noise_idx = 0
        
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
            tspan = np.arange(t0, t0+tmax-1e-4, h)
            sol = solve_ivp(eqns_std, (t0, t0+tmax),  y0, t_eval = np.arange(t0, t0+tmax-1e-4, h), events = event_func, rtol = 1, max_step=0.001, atol = 1)            
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
    
    def clamp(self, u, u_max = 30):
        u1, u2 = u[0], u[1]
        u1 = max(min(u1, u_max), -u_max) 
        u2 = max(min(u2, u_max), -u_max) 
        u = np.array([u1, u2])
        return u
    
    def control_ilqr(self, t, y, t0):

        t_step = int((t- t0)/self.dT)
        #print(t_step)
        if t_step >= len(self.ilqr.us):
            t_step = len(self.ilqr.us)-1

        if self.use_noise is True:
            q, dq = y[:3].copy(), y[3:].copy()
            q = q + self.q_noise[self.noise_idx]
            dq = dq + self.dq_noise[self.noise_idx]
            y = np.concatenate([q, dq])
            self.noise_idx += 1
            if self.noise_idx == len(self.q_noise):
                self.noise_idx = 0
        
        u = self.ilqr.us[t_step] - self.ilqr.K[t_step].dot(y - self.ilqr.xs[t_step])    # for crocoddyl    
        #u = self.ilqr.us[t_step] + self.ilqr.K[t_step].dot(y-self.ilqr.xs[t_step]) #for Teguh ilqr
        return self.clamp(u)
    
    def control_qp(self, t, y, t0):
        t_step = int((t- t0)/self.dT)
        if t_step >= len(self.xs_ref)-1:
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

        mode = 'numpy'
        M = self.eval_M(q, mode)
        C = self.eval_C(q, dq, mode)
        G = self.eval_G(q, mode)
        B = self.eval_B(mode)
        
        #use inverse dynamics instead
        tau = np.linalg.pinv(B).dot(M.dot(qdd) + C.dot(dq) + G)
        #return tau
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
    
    def eqns(self, t, y, t0,  parameters = None):    
        q = y[:3]
        dq = y[3:]
        
        mode = 'numpy'
        M = self.eval_M(q, mode)
        C = self.eval_C(q, dq, mode)
        G = self.eval_G(q, mode)
        B = self.eval_B(mode)
        
        
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
        mode = 'numpy'
        M = self.eval_M(q, mode)
        C = self.eval_C(q, dq, mode)
        G = self.eval_G(q, mode)
        B = self.eval_B(mode)
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
    value = z_swf + 0.01* cos(q[0]) + 0.0001
    return value
event_func.terminal = 1
event_func.direction = -1