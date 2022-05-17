import numpy as np
#import tensorflow as tf
from math import sqrt
import itertools

from numba import jit


import math
import random

norm = np.linalg.norm

@jit
def norm_grad(x):
    return x/norm(x)

@jit(nopython=True)
def sgd_debug(X,d,w,indices,schedule,t,tol):
    shuffle = np.random.shuffle
    n = len(X)
    yield X.copy()
    for epoch in range(len(schedule)):
        step = schedule[epoch]

        change = 10
        for i,j in indices:
            #old = np.linalg.norm(X[i]-X[j])
            #Get difference vector
            pq = X[i]-X[j]

            #Calculate its magnitude (numpys implementation was soooooo slowwwwww on my poor laptop at least)
            mag = (pq[0]*pq[0] + pq[1]*pq[1]) ** 0.5

            #derivative of eucldiean norm
            mag_grad = pq/mag

            #get the step size
            mu = (step*w[i][j]) / d[i][j] ** 2
            if mu >= 1: mu = 1

            #Find the distance to move both nodes as in https://github.com/jxz12/s_gd2
            r = (mu*(mag-d[i][j]))/(2*mag)
            stress = r*pq

            #repulsion step size and calculation
            # mu1 = step #* d[i][j] **2
            # if mu1 >= 0.5: mu1 = 0.5
            # repulsion = -mu1 * (mag_grad/(mag))

            #Normalize so the weights sum to 1
            l_sum = 1+t

            #Final gradient w.r.t X[i]
            m = stress

            #Update positions
            X[i] -= m
            X[j] += m

            #Track how much we changed
            newpq = X[i]-X[j]
            newmag = (newpq[0]*newpq[0] + newpq[1]*newpq[1]) ** 0.5
            change = max(change,abs(mag-newmag))


        if change < tol: break
        #print(get_cost(X,d,w,t))
        yield X.copy()
        shuffle(indices)
        #print("Epoch: " + str(epoch))
    return X

@jit(nopython=True,cache=True)
def sgd(X,d,w,indices,schedule,t,tol):
    shuffle = np.random.shuffle
    n = len(X)
    for epoch in range(len(schedule)):
        step = schedule[epoch]

        change = 10
        for i,j in indices:
            #old = np.linalg.norm(X[i]-X[j])
            #Get difference vector
            pq = X[i]-X[j]

            #Calculate its magnitude (numpys implementation was soooooo slowwwwww on my poor laptop at least)
            mag = np.sum(pq**2) ** 0.5

            #derivative of eucldiean norm
            mag_grad = pq/mag

            #get the step size
            mu = step / (d[i][j] ** 2)
            if mu >= 1: mu = 1

            #Find the distance to move both nodes as in https://github.com/jxz12/s_gd2
            r = (mu*(mag-d[i][j]))/(2*mag)
            stress = r*pq

            #repulsion step size and calculation
            # mu1 = (step)
            # if mu1 >= 1: mu1 = 1
            # repulsion = -mu1 * (mag_grad/(mag))

            #Normalize so the weights sum to 1
            l_sum = 1+t

            #Final gradient w.r.t X[i]
            m = stress

            #Update positions
            X[i] -= m
            X[j] += m

            #Track how much we changed
            newpq = X[i]-X[j]
            newmag = (newpq[0]*newpq[0] + newpq[1]*newpq[1]) ** 0.5
            change = max(change,abs(mag-newmag))


        if change < tol: break
        shuffle(indices)
        #print("Epoch: " + str(epoch))
    return X

def schedule_convergent(d,t_max,eps,t_maxmax):
    w = np.divide(np.ones(d.shape),d**2,out=np.zeros_like(d), where=d!=0)
    w_min,w_max = np.amin(w,initial=10000,where=w > 0), np.max(w)

    eta_max = 1.0 / w_min
    eta_min = eps / w_max

    lamb = np.log(eta_max/eta_min) / (t_max-1)

    # initialize step sizes
    etas = np.zeros(t_maxmax)
    eta_switch = 1.0 / w_max
    #print(eta_switch)
    for t in range(t_maxmax):
        eta = eta_max * np.exp(-lamb * t)
        if (eta < eta_switch): break

        etas[t] = eta

    tau = t
    for t in range(t,t_maxmax):
        eta = eta_switch / (1 + lamb*(t-tau))
        etas[t] = eta
        #etas[t] = 1e-7

    #etas = [eps for t in range(t_maxmax)]
    #print(etas)
    return np.array(etas)

class SGD:
    def __init__(self,dissimilarities,k=5,weighted=True,w = np.array([]), init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        self.d_min = 1
        self.n = len(self.d)
        if init_pos.any():
            self.X = np.asarray(init_pos)
        else: #Random point in the chosen geometry
            self.X = np.random.uniform(0,1,(self.n,2))

            #self.X = np.asarray(self.X)

        a = 1
        b = 1
        self.weighted = weighted
        if weighted:
            # self.w = set_w(self.d,k)
            self.w = w
        else:
            self.w = np.array([[ 1 if self.d[i][j] > 0 else 0 for i in range(self.n)]
                        for j in range(self.n)])

        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max

    def get_sched(self,num_iter):
        lamb = np.log(self.eta_min/self.eta_max)/(num_iter-1)
        sched = lambda count: self.eta_max*np.exp(lamb*count)
        #sched = lambda count: 1/np.sqrt(count+1)
        return np.array([sched(count) for count in range(100)])

    def solve(self,num_iter=60,debug=False,t=0.6,tol=1e-6,eps=0.1):
        import itertools

        indices = np.array(list(itertools.combinations(range(self.n), 2)))
        schedule = schedule_convergent(self.d,30,eps,num_iter)
        t = t if self.weighted else 0
        if debug:
            return [X for X in sgd_debug(self.X,self.d,self.w,indices,schedule,t,tol)]

        self.X = sgd(self.X,self.d,self.w,indices,schedule,t,tol)
        return self.X

    def solve_old(self,num_iter=1500,debug=False,t=1,radius=False, k=1,tol=1e-3):
        #import autograd.numpy as np
        from autograd import grad
        from sklearn.metrics import pairwise_distances
        import itertools

        indices = np.array(list(itertools.combinations(range(self.n), 2)))

        d = self.d
        w = self.w
        X = self.X
        N = len(X)

        if debug:
            hist = [np.ones(X.shape) for count in range(num_iter+1)]

        sizes = np.zeros(num_iter+1)
        movement = lambda X,X_c,step: np.sum(np.sum(X_c ** 2, axis=1) ** 0.5) / (N * step * np.max(np.max(X, axis=0) - np.min(X, axis=0)))

        eps = 1e-13
        epsilon = np.ones(d.shape)*eps
        def stress(X,t):                 # Define a function
            stress, l_sum = 0, 1+t


            #Stress
            ss = (X * X).sum(axis=1)
            diff = ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)
            diff = np.sqrt(np.maximum(diff,epsilon))
            stress = np.sum( w * np.square(d-diff) )

            #repulsion
            diff = diff + eps

            # condlist = [diff<10, diff>=10]
            # choicelist = [np.log(diff), 0]
            # r = np.select(condlist, choicelist, 0)
            # r = -np.sum( r )
            r = -np.sum( np.log(diff+eps) )

            return (1/l_sum) * np.sum(stress) + (t/l_sum) * r

        step,change,momentum = 0.001, 0.0, 0.5
        #grad_stress = jit(grad(pair_stress))
        cost = 0

        #t = 0.6
        schedule = schedule_convergent(d,30,0.1,200)
        for epoch in range(len(schedule)):
            step = schedule[epoch]
            #step = step if step < 0.5 else 0.5


            X,change = iteration(X,w,d,indices,t,step)


            # x_prime = grad_stress(X,t)
            #
            # new_change = step * x_prime + momentum * change
            #
            # X -= new_change
            #
            # if abs(new_change-change).max() < 1e-3: momentum = 0.8
            sizes[epoch] = change
            #
            # change = new_change

            if epoch > 15:
                max_change = change
                print("Epoch: {} . Max change over last 1: {} . ".format(epoch,round(change,5),end='\r') )
                #if max_change < tol: break
                # if epoch % 101 == 0:
                #     if stress(X,t) < -70000: break

            #print(stress(X,t))
            #self.X = X
            if debug:
                hist[epoch] = X.copy()
        self.X = X
        print()
        if debug:
            return hist
        return X.copy()



    def compute_step_size(self,count,num_iter):
        #return 1/pow(5+count,1)

        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return np.random.uniform(0,1,2)


def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product




def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new


def set_w(d,k):
    f = np.zeros(d.shape)
    for i in range(len(d)):
        for j in range(len(d)):
            if i == j:
                f[i][j] = 100000
            else:
                f[i][j] = d[i][j]
    f += np.random.normal(scale=0.1,size=d.shape)
    k_nearest = [get_k_nearest(f[i],k) for i in range(len(d))]

    #1/(10*math.exp(d[i][j]))
    w = np.asarray([[ 1e-5 if i != j else 0 for i in range(len(d))] for j in range(len(d))])
    for i in range(len(d)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1
                w[j][i] = 1

    return w
