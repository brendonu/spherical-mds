import networkx as nx
import numpy as np
#import igraph as ig
import matplotlib.pyplot as plt
#import tensorflow as tf
#import drawSvg as draw
from math import sqrt
import sys
import itertools


import math
import random
import cmath
import copy
import time
import os
from numba import jit

def sphere_dist(xi,xj):
    p1,l1 = xi
    p2,l2 = xj
    return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))


@jit(nopython=True,cache=True)
def gradient(p,q):
    sin, cos = np.sin, np.cos
    rt = lambda x: pow(x,0.5)

    y,x = p
    b,a = q

    denom = 1/rt(1-pow((sin(b)*sin(y) + cos(b)*cos(y)*cos(a-x)),2))

    dx = -denom * (sin(a-x)*cos(b)*cos(y))
    dy = denom * (-sin(b)*cos(y) + sin(y)*cos(b)*cos(a-x))

    da = -dx
    db = denom * (sin(b)*cos(y)*cos(a-x) - sin(y)*cos(b))

    return np.array([[dy,dx],
                     [db,da]])

@jit(nopython=True,cache=True)
def solve_stochastic(d,indices,
                    w=None,num_iter=100, epsilon=1e-5,debug=False,schedule='fixed',init_pos=None,
                    switch_step=30,steps=None):

    # from autograd import grad
    # import autograd.numpy as np
    n = d.shape[0]

    #Initialize positions
    if init_pos:
        X = init_pos
    else:
        x1 = np.random.uniform(0, math.pi, (n,1) )
        x2 = np.random.uniform(0,2*math.pi, (n,1) )
        X = np.concatenate( (x1,x2), axis=1 )

    w = w if w else np.ones( (n,n) )

    #grab learning rate
    #steps = schedule_convergent(d,switch_step,epsilon,num_iter)
    #steps = np.ones(num_iter) * 0.001
    max_change, shuffle,cap = 0, random.shuffle, 0.5
    tol = np.ones( (n,n) ) * 1e-13

    shuffle(indices)

    sin, cos, asin, acos, sqrt = np.sin, np.cos, np.arcsin, np.arccos, np.sqrt

    # def sphere_stress(X):
    #
    #     lat = X[:,0]
    #     lng = X[:,1]
    #     diff_lat = lat.reshape((n,1)) - lat
    #     diff_lng = lng.reshape((n,1)) - lng
    #     diff = sin(diff_lat/2)**2 + cos(lat.reshape((n,1)))*cos(lat) * sin(diff_lng/2)**2
    #     Y =  2 * asin(sqrt(np.maximum(diff,tol)))
    #     residual = (Y-d) ** 2
    #     return residual.sum() / (n**2)

    geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )

    def stress(X):
        stress = 0
        for i in range(n):
            for j in range(i):
                stress += w[i][j]*pow(geodesic(X[i],X[j])-d[i][j],2)
        return stress / pow(n,2)


    error = lambda x1,x2, dij: (geodesic(x1,x2) - dij) **2
    #haver_grad = grad(error)
    prev_error, now_error = 0,0
    for step in steps:

        for i,j in indices:
            wc =  step / (d[i][j]**2)
            wc = cap if wc > cap else wc

            #gradient
            g = gradient(X[i],X[j]) * 2 * (geodesic(X[i],X[j])-d[i][j])
            m = wc*g


            X[i] = X[i] - m[0]
            X[j] = X[j] - m[1]

        shuffle(indices)
        now_error = stress(X)
        if abs(now_error-prev_error) < epsilon:
            break
        prev_error = now_error
        if debug:
            print(stress(X))


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

class SMDS:
    def __init__(self,dissimilarities,init_pos=np.array([]),scale_heuristic=True):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        if scale_heuristic:
            self.d *= (math.pi/self.d_max)
        self.d_min = 1
        self.n = len(self.d)



    def solve(self,num_iter=500,epsilon=1e-3,debug=False,schedule='fixed'):
        steps = schedule_convergent(self.d,30,0.01,num_iter)
        X = solve_stochastic(self.d,np.array( list(itertools.combinations(range(self.n) , 2) )),
                            w=None,num_iter=num_iter,steps=steps,debug=debug)
        self.X = X
        return X


    def calc_stress(self):
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return stress

    def calc_distortion(self):
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion







def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag

def geodesic(xi,xj):
    return sphere_dist(xi,xj)

def sphere_dist(xi,xj):
    sin, cos = np.sin, np.cos
    p1,l1 = xi
    p2,l2 = xj
    return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product
