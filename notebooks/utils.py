from numpy.linalg import inv
import numpy as np
import time
import pyscreenshot as ImageGrab

def extract_init_goal(sln):
    #extract initial and goal configs at the last step of sln
    Y = np.array(sln['Y'])
    y0 = []
    yT = []
    for i in range(len(Y)):
        y0 += [Y[i][0]]
        yT += [Y[i][-1]]
    y0 = np.array(y0)
    yT = np.array(yT)
    return y0, yT

def set_ref(ddp, xs):
    #given the reference traj. xs, set the ddp to follow this xs
    T = len(xs)
    for i in range(T-1):
        ddp.problem.runningModels[i].cost_model.costs[0].set_ref(xs[i])
    ddp.problem.terminalModel.cost_model.costs[0].set_ref(xs[-1])
    
def set_Qref(ddp, Qref):
    #given the precision matrices Qref, set the ddp to use Qref in the cost function
    T = len(Qref)
    for i in range(T-1):
        ddp.problem.runningModels[i].cost_model.costs[0].Q = Qref[i]
    ddp.problem.terminalModel.cost_model.costs[0].Q = Qref[T-1]
    
def extract_ref(mu, sigma, Dx, T_hor, x, reg = 1e-6):
    #given the distribution N(mu,sigma), obtain the reference traj. distribution
    # as the marginal distribution at time t
    mu_ = mu.reshape(-1, Dx)
    T = mu.shape[0]
    
    #obtain the reference trajectory
    ref_x = subsample(np.vstack([x[None,:], mu_[:T_hor]]), T_hor+1)
    
    #obtain the reference precision
    Qs = np.zeros((T_hor+1, Dx, Dx))
    if T_hor < T:
        #if the horizon is within the remaining time steps, extract the marginal distribution
        for i in range(T_hor):
            Qs[i+1] = inv(sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
    else:
        #if the horizon exceeds the remaining time steps
        for i in range(T):
            Qs[i+1] = inv(sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
        for i in range(T, T_hor):
            Qs[i+1] = Qs[T]

    #the first Q does not affect the OCP and can be set as anything
    Qs[0] = Qs[1].copy()

    return ref_x, Qs

def calc_detail_cost(xs, us, ddp):
    rmodels = ddp.problem.runningModels
    cost_control = 0.
    cost_goal = 0.
    cost_state = 0.
    cost_col = []
    for i in range(len(xs)-1):
        costs = rmodels[i].cost_model.costs
        cost_state += costs[0].calc(xs[i], us[i])  
        cost_control +=  costs[1].calc(xs[i], us[i])
        cost_col += [costs[2].calc(xs[i], us[i])]
    cost_goal = ddp.problem.terminalModel.cost_model.calc(xs[-1], us[-1])
    return  cost_state, cost_control, cost_col, cost_goal

def lin_interpolate(state1, state2, n=1.):
    state_list = []
    for i in range(n+1):
        state_list.append(state1 + 1.*i*(state2-state1)/n)
    return state_list

def subsample(X,N):
    '''Subsample in N iterations the trajectory X. The output is a 
    trajectory similar to X with N points. '''
    nx  = X.shape[0]
    idx = np.arange(float(N))/(N-1)*(nx-1)
    hx  = []
    for i in idx:
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        di = i%1
        x  = X[i0,:]*(1-di) + X[i1,:]*di
        hx.append(x)
    return np.vstack(hx)

def save_screenshot(x,y,w,h,file_name, to_show='False'):
    # part of the screen
    im=ImageGrab.grab(bbox=(x,y,w,h))
    if to_show:
        im.show()
    # save to file
    im.save(file_name)
    return im

def normalize(x):
    return x/np.linalg.norm(x)

