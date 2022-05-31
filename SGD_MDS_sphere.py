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
def gradient(p,q,r):
    sin, cos = np.sin, np.cos
    rt = lambda x: pow(x,0.5)

    y,x = p
    b,a = q

    denom = 1/rt(1-pow((sin(b)*sin(y) + cos(b)*cos(y)*cos(a-x)),2))

    dx = -denom * (sin(a-x)*cos(b)*cos(y))
    dy = denom * (-sin(b)*cos(y) + sin(y)*cos(b)*cos(a-x))

    da = -dx
    db = denom * (sin(b)*cos(y)*cos(a-x) - sin(y)*cos(b))

    return r * np.array([[dy,dx],
                     [db,da]])




@jit(nopython=True,cache=True)
def solve_stochastic(d,indices,
        w=None,num_iter=100, epsilon=1e-5,debug=False,schedule='fixed',init_pos=None,
        switch_step=30,lr_cap=0.5):

    n = d.shape[0]
    r = 1

    steps = schedule_convergent(d,30,0.01,num_iter)

    #Initialize positions
    if init_pos:
        X = init_pos
    else:
        x1 = np.random.uniform(0, math.pi, (n,1) )
        x2 = np.random.uniform(0,2*math.pi, (n,1) )
        X = np.concatenate( (x1,x2), axis=1 )

    w = w if w else np.ones( (n,n) )

    max_change, shuffle,cap = 0, random.shuffle, lr_cap
    tol = np.ones( (n,n) ) * 1e-13

    shuffle(indices)

    sin, cos, asin, acos, sqrt = np.sin, np.cos, np.arcsin, np.arccos, np.sqrt

    # def geodesic(x1,x2):
    #     return acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )

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
            delta = geodesic(X[i],X[j])
            g = gradient(X[i],X[j],r) * 2 * (r*delta-d[i][j])
            m = wc*g


            X[i] = X[i] - m[0]
            X[j] = X[j] - m[1]
            #r = r - wc*  2*delta*(r*delta - d[i][j])
            #if r <= 0: r = epsilon

        shuffle(indices)
        now_error = stress(X)
        if abs(now_error-prev_error) < epsilon:
            break
        prev_error = now_error
        if debug:
            print(stress(X))


    return X


@jit(nopython=True,cache=True)
def schedule_convergent(d,t_max,eps,t_maxmax):
    w = d.copy()
    w_min,w_max = 10000, 0
    for i in range(w.shape[0]):
        for j in range(w.shape[0]):
            if i == j:
                w[i,j] = 0
            else:
                w[i,j] = d[i][j] ** -2
                w_min = min(w[i,j], w_min)
                w_max = max(w[i,j],w_max)

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
    return etas

@jit(nopython=True,cache=True)
def solve_stochastic_debug(d,indices,
        w=None,num_iter=100, epsilon=1e-5,debug=False,schedule='fixed',init_pos=None,
        switch_step=30,lr_cap=0.5):

    n = d.shape[0]
    r = 1
    hist = []
    r_hist = []

    if schedule=='fixed':
        steps = np.ones(num_iter)*1e-3
    elif schedule=='convergent':
        steps = schedule_convergent(d,30,0.01,num_iter)
    elif schedule=='sqrt':
        steps = np.zeros(num_iter)
        for i in range(num_iter):
            steps[i] = 1/np.sqrt(i+1)

    #Initialize positions
    if init_pos:
        X = init_pos
    else:
        x1 = np.random.uniform(0, math.pi, (n,1) )
        x2 = np.random.uniform(0,2*math.pi, (n,1) )
        X = np.concatenate( (x1,x2), axis=1 )

    w = w if w else np.ones( (n,n) )

    max_change, shuffle,cap = 0, random.shuffle, lr_cap
    tol = np.ones( (n,n) ) * 1e-13

    shuffle(indices)

    sin, cos, asin, acos, sqrt = np.sin, np.cos, np.arcsin, np.arccos, np.sqrt

    # def geodesic(x1,x2):
    #     return acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )

    geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )

    def stress(X,r):
        stress = 0
        for i in range(n):
            for j in range(i):
                stress += w[i][j]*pow(r*geodesic(X[i],X[j])-d[i][j],2)
        return stress / pow(n,2)


    error = lambda x1,x2, dij: (geodesic(x1,x2) - dij) **2
    #haver_grad = grad(error)
    prev_error, now_error = stress(X,r),0

    for step in steps:

        for i,j in indices:
            wc =  step / (d[i][j]**2)
            wc = cap if wc > cap else wc

            #gradient
            delta = geodesic(X[i],X[j])
            g = gradient(X[i],X[j],r) * 2 * (r*delta-d[i][j])
            m = wc*g


            X[i] = X[i] - m[0]
            X[j] = X[j] - m[1]
            #r = r - wc*  2*delta*(r*delta - d[i][j])
            #if r <= 0: r = epsilon

        shuffle(indices)
        now_error = stress(X,r)
        hist.append(X.copy())
        r_hist.append(r)
        if abs(now_error-prev_error)/prev_error < epsilon:
            break
        prev_error = now_error
        if debug:
            print(stress(X,r))


    return hist, r_hist

class SMDS:
    def __init__(self,dissimilarities,init_pos=np.array([]),scale_heuristic=True):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        if scale_heuristic:
            self.d *= (math.pi/self.d_max)
        self.d_min = 1
        self.n = len(self.d)



    def solve(self,num_iter=500,epsilon=1e-3,debug=False,schedule='fixed',cap=0.5):
        steps = schedule_convergent(self.d,30,0.01,num_iter)
        if debug:
            return solve_stochastic_debug(self.d,np.array( list(itertools.combinations(range(self.n) , 2) )),schedule=schedule, w=None,num_iter=num_iter,debug=debug,lr_cap=cap)
        X = solve_stochastic(self.d,np.array( list(itertools.combinations(range(self.n) , 2) )),
                            w=None,num_iter=num_iter,debug=debug,lr_cap=cap)
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
