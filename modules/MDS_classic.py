import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from math import sqrt


import math
import random
import cmath
import copy
import time
import os


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

class MDS:
    def __init__(self,dissimilarities,geometry='euclidean',init_pos=np.array([])):
        self.d = scale_matrix(dissimilarities,math.pi)
        #self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        self.d_min = 1
        self.n = len(self.d)
        self.geometry = geometry
        if init_pos.any():
            self.X = np.asarray(init_pos)
        else: #Random point in the chosen geometry
            self.X = np.zeros((len(self.d),2))
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X)

        self.w = [[ 1/pow(self.d[i][j],2) if i != j else 0 for i in range(self.n)]
                    for j in range(self.n)]
        for i in range(len(self.d)):
            self.w[i][i] = 0

        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/self.w_max

        self.history = []
        self.pos_history = []

    def solve(self,num_iter=100,epsilon=1e-3,debug=False):
        import autograd.numpy as np
        from autograd import grad
        from sklearn.metrics import pairwise_distances

        sin, cos, asin, sqrt = np.sin, np.cos, np.arcsin, np.sqrt


        current_error,error,step,count = 1000,1,1,0
        prev_error = 1000000

        steps = schedule_convergent(self.d,15,0.01,200)

        X,d,w = self.X, self.d, self.w
        n = len(X)
        tol = np.ones( (n,n) )* 1e-13

        def sphere_stress(X):

            lat = X[:,0]
            lng = X[:,1]
            diff_lat = lat[:,None] - lat
            diff_lng = lng[:,None] - lng
            diff = sin(diff_lat/2)**2 + cos(lat[:,None])*cos(lat) * sin(diff_lng/2)**2
            Y =  2 * asin(sqrt(np.maximum(diff,tol)))
            residual = (Y-d) ** 2
            return residual.sum() / (n**2)

        step,change,momentum = 10, 0.0, 0.5
        grad_stress = grad(sphere_stress)
        cost = 0

        for epoch in range(num_iter):
            x_prime = grad_stress(X)

            new_change = step * x_prime + momentum * change

            X -= new_change

            #if abs(new_change-change).max() < 1e-3: momentum = 0.8
            if abs(new_change-change).max() < 1e-5: break
            #sizes[epoch] = movement(X,new_change,step)

            change = new_change

            # step = steps[epoch] if epoch < len(steps) else steps[-1]
            # step = 1/(np.sqrt(epoch + 1))
            if epoch % 500 == 0 and epoch > 1:
                step = step /10
            if debug:
                print(sphere_stress(X))
            # self.history.append(sphere_stress(X))
            #self.X = X
        self.X = X
        return X,epoch,self.history


    def geodesic(self,xi,xj):
        if self.geometry == 'euclidean':
            return euclid_dist(xi,xj)
        elif self.geometry == 'spherical':
            return sphere_dist(xi,xj)
        elif self.geometry == 'hyperbolic':
            return polar_dist(xi,xj)

    def grad(self,xi,xj):
        if self.geometry == 'euclidean':
            return euclid_grad(xi,xj)
        elif self.geometry == 'spherical':
            return sphere_grad(xi,xj)
        elif self.geometry == 'hyperbolic':
            return hyper_grad(xi,xj)

    def calc_stress2(self,X0):
        stress = 0
        for i in range(len(self.d)):
            for j in range(i):
                stress += (1/tf.math.cosh(self.d[i][j]))*tf.math.cosh(self.d[i][j]-self.geodesic(X0[i],X0[j]))
        return stress

    def calc_stress(self):
        stress = 0
        for i in range(len(self.d)):
            for j in range(i):
                stress += self.w[i][j]*pow(self.geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return stress

    def calc_stress1(self,X0,wrt,i):
        stress = 0
        for j in range(len(self.d)):
            if i != j:
                stress += (1/pow(self.d[i][j],2))*pow(self.geodesic(wrt,X0[j])-self.d[i][j],2)
        return stress

    def calc_distortion(self):
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((self.geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion

    def compute_step_size(self,count,num_iter):
        # lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        # return self.eta_max*math.exp(lamb*count)
        a = 1
        b = 1#-math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return a/(pow(1+b*count,0.5))


    def init_point(self):
        if self.geometry == 'euclidean':
            return [random.uniform(-1,1),random.uniform(-1,1)]
        elif self.geometry == 'spherical': #Random point on sphere
            return [random.uniform(0,math.pi),random.uniform(0,2*math.pi)]
        elif self.geometry == 'hyperbolic': #Random point on unit (hyperbolic) circle
            r = pow(random.uniform(0,.5),0.5)
            theta = random.uniform(0,2*math.pi)
            x = math.atanh(math.tanh(r)*math.cos(theta))
            y = math.asinh(math.sinh(r)*math.sin(theta))
            return [r,theta]

def euclid_grad(p,q):
    return (p-q)/euclid_dist(p,q)

def hyper_grad(p,q):
    r,t = p
    a,b = q
    sin = np.sin
    cos = np.cos
    sinh = np.sinh
    cosh = np.cosh
    bottom = 1/pow(pow(part_of_dist_hyper(p,q),2)-1,0.5)
    if np.isinf(bottom):
        bottom = 1000

    #delta_a = -(cos(b-t)*sinh(r)*cosh(a)-sinh(a)*cosh(r))*bottom
    #delta_b = (sin(b-t)*sinh(a)*sinh(r))*bottom

    delta_r = -1*(cos(b-t)*sinh(a)*cosh(r)-sinh(r)*cosh(a))*bottom
    delta_t = -1*(sin(b-t)*sinh(a)*sinh(r))*bottom

    return np.array([delta_r,delta_t])

def part_of_dist_hyper(xi,xj):
    r,t = xi
    a,b = xj
    sinh = np.sinh
    cosh = np.cosh
    #print(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    return np.cos(b-t)*sinh(a)*sinh(r)-cosh(a)*cosh(r)

def sphere_grad(p,q):
    lamb1,phi1 = p
    lamb2,phi2 = q
    bottom = 1/pow(1-pow(part_of_dist_sphere(p,q),2),0.5)
    if np.isinf(bottom):
        bottom = 1000
    sin = np.sin
    cos = np.cos
    x = -sin(lamb2-lamb1)*cos(phi2)*cos(phi1)*bottom
    y = (-sin(phi2)*cos(phi1) + sin(phi1)*cos(phi2)*cos(lamb2-lamb1))*bottom
    return np.array([x,y])

def part_of_dist_sphere(xi,xj):
    lamb1,phi1 = xi
    lamb2,phi2 = xj
    sin = np.sin
    cos = np.cos
    return sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(lamb2-lamb1)

def polar_dist(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return np.arccosh(np.cosh(r1)*np.cosh(r2)-np.sinh(r1)*np.sinh(r2)*np.cos(theta2-theta1))

def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag

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
        myList.append(distance[x])

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

def chord_dist(xi,xj):
    return pow(2 - 2*tf.math.sin(xi[1])*tf.math.sin(xj[1])*tf.math.cos(abs(xi[0]-xj[0]))- 2*tf.math.cos(xi[1])*tf.math.cos(xj[1]),0.5)

def sphere_dist(xi,xj):
    sin, cos = np.sin, np.cos
    l1, p1 = xi
    l2, p2 = xj
    return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))


def euclid_dist(x1,x2):
    x = x2[0]-x1[0]
    y = x2[1]-x1[1]
    return pow(x*x+y*y,0.5)

def hyperbolic_dist(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return tf.math.acosh(tf.math.cosh(r1)*tf.math.cosh(r2)-tf.math.sinh(r1)*tf.math.sinh(r2)*tf.math.cos(theta2-theta1))

def poincare_dist(x,y):
    A = euclid_dist(x,y)
    B = (1-pow(euclid_dist([0,0],x),2))
    C = (1-pow(euclid_dist([0,0],y),2))
    dist = tf.math.acosh(1+2*(A)/(B*C))

    return dist

def lob_dist(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    return tf.math.acosh(tf.math.cosh(y1)*tf.math.cosh(x2-x1)*tf.math.cosh(y2)-tf.math.sinh(y1)*tf.math.sinh(y2))

def hyperbolic_dist_old(p,q):
    op = euclid_dist([0,0],p)
    oq = euclid_dist([0,0],q)
    pq = euclid_dist(p,q)
    return tf.math.acosh(1+(2*pow(pq,2)*1)/(1-pow(op,2))*(1-pow(oq,2)))

def output_sphere(G,X):
    def latLongToCart(thetas):
        phi = thetas[1]
        theta = thetas[0]
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        return np.array([x,y,z])

    count = 0
    for x in G.nodes():
        G.nodes[x]['pos'] = str(X[count][1]) + "," + str(X[count][0])
        dim3 = latLongToCart(X[count])
        G.nodes[x]['dim3pos'] = str(dim3[0]) + "," + str(dim3[1]) + "," + str(dim3[2])
        lng = X[1]*(180.0/math.pi)-180
        lat = X[0]*(180.0/math.pi)
        #print((lng,lat))
        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output_sphere.dot")

def output_hyperbolic(G,X):
    count = 0
    for i in G.nodes():
        x,y = X[count]
        #Rh = np.arccosh(np.cosh(x)*np.cosh(y))
        #theta = 2*math.atan2(np.sinh(x)*np.cosh(y)+pow(pow(np.cosh(x),2)*pow(np.cosh(y),2)-1,0.5),np.sinh(y))
        #Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)
        G.nodes[i]['mypos'] = str(x) + "," + str(y)
        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output_hyperbolic.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/old/jsCanvas/graphs/hyperbolic_colors.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/maps/static/graphs/hyperbolic_colors.dot")

def output_euclidean(G,X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos,with_labels=True)
    plt.show()
    plt.clf()

def all_three(d):
    stresses = {}

    Y = MDStf(d,'euclidean')
    X = Y.solve()
    stresses['euclidean'] = float(Y.calc_distortion())
    output_euclidean(X)

    Y = MDStf(d,'spherical')
    X = Y.solve()
    stresses['spherical'] = float(Y.calc_distortion())
    output_sphere(X)

    Y = MDStf(d,'hyperbolic')
    X = Y.solve()
    stresses['hyperbolic'] = float(Y.calc_distortion())
    output_hyperbolic(X)

    print(stresses)

def get_distortion(X,d):
    distortion = 0
    for i in range(d.shape[0]):
        for j in range(i):
            distortion += abs((sphere_dist(X[i],X[j])-d[i][j]))/d[i][j]
    return (1/choose(d.shape[0],2))*distortion

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product
