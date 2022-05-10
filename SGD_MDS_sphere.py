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
                    w=None,num_iter=100, epsilon=1e-3,debug=False,schedule='fixed',init_pos=None,
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
    steps = np.ones(num_iter) * 0.001
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

    error = lambda x1,x2, dij: (geodesic(x1,x2) - dij) **2
    #haver_grad = grad(error)

    for step in steps:

        for i,j in indices:
            wc =  step / (d[i][j]**2)
            wc = cap if wc > cap else wc

            #gradient
            g = gradient(X[i],X[j]) * 2 * geodesic(X[i],X[j])
            m = wc*g


            X[i] = X[i] - m[0]
            X[j] = X[j] - m[1]

        shuffle(indices)
        print(step)


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
    print(eta_switch)
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
    def __init__(self,dissimilarities,init_pos=np.array([])):
        self.d = scale_matrix(dissimilarities,math.pi)
        #self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        #self.d = (math.pi/self.d_max) * dissimilarities
        self.d_min = 1
        self.n = len(self.d)
        if init_pos.any():
            self.X = np.asarray(init_pos)
        else: #Random point in the chosen geometry
            self.X = np.zeros((len(self.d),2))
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X)

        self.w = [[ 1/pow(self.d[i][j],2) if i != j else 0 for i in range(self.n)]
                    for j in range(self.n)]


        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max

        self.history = []
        self.pos_history = []

    def solve(self,num_iter=15,epsilon=1e-3,debug=False,schedule='fixed'):
        steps = schedule_convergent(self.d,30,0.01,num_iter)
        X = solve_stochastic(self.d,np.array( list(itertools.combinations(range(self.n) , 2) )),
                            w=None,num_iter=num_iter,steps=steps)
        self.X = X
        return X

    def get_step(self,count,num_iter,schedule):
        if schedule == 'fixed':
            return 0.01
        elif schedule == 'exp':
            return self.compute_step_size(count,num_iter)
        elif schedule == 'frac':
            return self.compute_step_size_frac(count,num_iter)
        elif schedule == 'sqrt':
            return self.compute_step_size_sqrt(count,num_iter)

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

    def calc_gradient(self,i,j):
        X0 = tf.Variable(self.X)
        with tf.GradientTape() as tape:
            Y = self.calc_stress(X0)
        dy_dx = tape.gradient(Y,X0).numpy()
        #dy = dy_dx.numpy()
        for i in range(len(self.d)):
            dy_dx[i] = normalize(dy_dx[i])
        return dy_dx

    def compute_step_size(self,count,num_iter):


        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)

    def compute_step_size_frac(self,count,num_iter):
        a = 1/self.w_max
        b = -math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return a/pow(1+b*count,1)

    def compute_step_size_sqrt(self,count,num_iter):
        a = 1/self.w_max
        b = -math.log(self.eta_min/self.eta_max)/(15-1)
        return a/(pow(1+b*count,0.5))


    def init_point(self):
        return [random.uniform(0,2*math.pi),random.uniform(0,math.pi)]




def pair_stress(p,q,t):
    sq = lambda x: pow(x,2)
    return sq(geodesic(p,q) - t)

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


def bfs(G,start):
    queue = [start]
    discovered = [start]
    distance = {start: 0}

    while len(queue) > 0:
        v = queue.pop()

        for w in G.neighbors(v):
            if w not in discovered:
                discovered.append(w)
                distance[w] =  distance[v] + 1
                queue.insert(0,w)

    myList = []
    for x in G.nodes:
        if x in distance:
            myList.append(distance[x])
        else:
            myList.append(len(G.nodes)+1)

    return myList

def all_pairs_shortest_path(G):
    d = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]

    count = 0
    for node in G.nodes:
        d[count] = bfs(G,node)
        count += 1
    return d

def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new


def euclid_dist(x1,x2):
    x = x2[0]-x1[0]
    y = x2[1]-x1[1]
    return pow(x*x+y*y,0.5)


def output_euclidean(G,X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos)
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.show()
    plt.clf()

    count = 0
    for i in G.nodes():
        x,y = X[count]
        G.nodes[i]['pos'] = str(100*x) + "," + str(100*y)

        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output.dot")

def output_sphere(G,X,fname="output_sphere.dot"):
    def latLongToCart(thetas):
        phi = thetas[1]
        theta = thetas[0]
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        return np.array([x,y,z])

    import graph_tool.all as gt

    Z = gt.Graph(directed=False)
    Z.add_vertex(G.num_vertices())
    Z.add_edge_list(G.edges())

    Z.save('temp.xml',fmt="graphml")
    G = nx.read_graphml('temp.xml')

    count = 0
    for x in G.nodes():
        G.nodes[x]['pos'] = str(X[count][0]) + "," + str(X[count][1])
        dim3 = latLongToCart(X[count])
        G.nodes[x]['dim3pos'] = str(dim3[0]) + "," + str(dim3[1]) + "," + str(dim3[2])
        lng = X[1]*(180.0/math.pi)-180
        lat = X[0]*(180.0/math.pi)
        #print((lng,lat))
        count += 1
    nx.drawing.nx_agraph.write_dot(G, fname)

#Code start

#G = nx.triangular_lattice_graph(5,5)
#G = nx.drawing.nx_agraph.read_dot('input.dot')
#G = nx.grid_graph([5,5],periodic=True)
#G = nx.full_rary_tree(2,100)
#G = nx.hypercube_graph(3)
#G = nx.read_graphml('newayo.xml')
#G = nx.random_geometric_graph(25,0.5)
#d = np.array(all_pairs_shortest_path(G))/1

#Y = myMDS(d)
#Y.solve(300,debug=True)
#print(Y.calc_distortion())
#output_sphere(Y.X)
