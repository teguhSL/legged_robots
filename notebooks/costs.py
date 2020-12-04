import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv

class CostModelQuadratic():
    def __init__(self, sys, Q = None, R = None, x_ref = None, u_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.Q, self.R = Q, R
        if Q is None: self.Q = np.zeros((self.Dx,self.Dx))
        if R is None: self.R = np.zeros((self.Du,self.Du))
        self.x_ref, self.u_ref = x_ref, u_ref
        if x_ref is None: self.x_ref = np.zeros(self.Dx)
        if u_ref is None: self.u_ref = np.zeros(self.Du)
            
    def set_ref(self, x_ref=None, u_ref=None):
        if x_ref is not None:
            self.x_ref = x_ref
        if u_ref is not None:
            self.u_ref = u_ref
    
    def calc(self, x, u):
        self.L = 0.5*(x-self.x_ref).T.dot(self.Q).dot(x-self.x_ref) + 0.5*(u-self.u_ref).T.dot(self.R).dot(u-self.u_ref)
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = self.Q.dot(x-self.x_ref)
        self.Lu = self.R.dot(u-self.u_ref)
        self.Lxx = self.Q.copy()
        self.Luu = self.R.copy()
        self.Lxu = np.zeros((self.Dx, self.Du))
        
class CostModelSum():
    def __init__(self, sys, costs):
        self.sys = sys
        self.costs = costs
        self.Dx, self.Du = sys.Dx, sys.Du
    
    def calc(self, x, u):
        self.L = 0
        for i,cost in enumerate(self.costs):
            cost.calc(x, u)
            self.L += cost.L
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = np.zeros(self.Dx)
        self.Lu = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx,self.Dx))
        self.Luu = np.zeros((self.Du,self.Du))
        self.Lxu = np.zeros((self.Dx,self.Du))
        for i,cost in enumerate(self.costs):
            cost.calcDiff(x, u)
            self.Lx += cost.Lx
            self.Lu += cost.Lu
            self.Lxx += cost.Lxx
            self.Luu += cost.Luu
            self.Lxu += cost.Lxu
            
class CostModelQuadraticTranslation():
    '''
    The quadratic cost model for the end effector, p = f(x)
    '''
    def __init__(self, sys, W, p_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.W = W
        self.p_ref = p_ref
        if p_ref is None: self.p_ref = np.zeros(3)
            
    def set_ref(self, p_ref):
        self.p_ref = p_ref
        
    def calc(self, x, u):
        x,z,_,_ = self.sys.kin_swf(x[:3], x[3:])
        p = np.array([x,z])
        self.L = 0.5*(p-self.p_ref).T.dot(self.W).dot(p-self.p_ref) 
        return self.L
    
    def calcDiff(self, x, u):
        self.J   = self.sys.compute_Jacobian_swf(x[:3], x[3:])
        x,z,_,_      = self.sys.kin_swf(x[:3], x[3:])
        p = np.array([x,z])        
        self.Lx  = self.J.T.dot(self.W).dot(p-self.p_ref)
        self.Lx = np.concatenate([self.Lx, np.zeros(self.Dx/2)])
        self.Lu  = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx, self.Dx))
        self.Lxx[:self.Dx/2, :self.Dx/2] = self.J.T.dot(self.W).dot(self.J)
        self.Luu = np.zeros((self.Du, self.Du))
        self.Lxu = np.zeros((self.Dx, self.Du))
        
class CostModelQuadraticLinVel():
    '''
    The quadratic cost model for the end effector, p = f(x)
    '''
    def __init__(self, sys, W, p_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.W = W
        self.p_ref = p_ref
        if p_ref is None: self.p_ref = np.zeros(3)
            
    def set_ref(self, p_ref):
        self.p_ref = p_ref
        
    def calc(self, x, u):
        x,z,dx,dz = self.sys.kin_hip(x[:3], x[3:])
        p = np.array([dx,dz])
        self.L = 0.5*(p-self.p_ref).T.dot(self.W).dot(p-self.p_ref) 
        return self.L
    
    def calcDiff(self, x, u):
        self.J   = self.sys.compute_Jacobian_vhip(x[:3], x[3:])
        x,z,dx,dz      = self.sys.kin_hip(x[:3], x[3:])
        p = np.array([dx,dz])        
        self.Lx  = self.J.T.dot(self.W).dot(p-self.p_ref)
        self.Lu  = np.zeros(self.Du)
        self.Lxx = self.J.T.dot(self.W).dot(self.J)
        self.Luu = np.zeros((self.Du, self.Du))
        self.Lxu = np.zeros((self.Dx, self.Du))