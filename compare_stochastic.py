import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS
from SGD_MDS2 import SGD
from graph_functions import *
import matplotlib.pyplot as plt
from recursive_divide import subdivide_graph
import math

import graph_tool.all as gt

# def sphere_dist(xi,xj):
#     sin, cos = np.sin, np.cos
#     l1, p1 = xi
#     l2, p2 = xj
#     return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))

def get_stress(X,d):
    stress = 0
    for i in range(len(d)):
        for j in range(i):
            stress += ((d[i][j] - sphere_dist(X[i],X[j])) ** 2 ) / d[i][j] ** 2
    return stress

def generate_stress_plots():
    G = gt.load_graph('outputs/bad_example.dot')
    #G = gt.complete_graph(4)
    d = get_distance_matrix(G,verbose=False)

    n = 30

    avg_fixed = np.zeros(200)
    avg_frac = np.zeros(200)
    avg_exp = np.zeros(200)
    avg_sqrt = np.zeros(200)

    for i in range(n):
        fixed = SMDS(d)
        fixed.solve(15,debug=True,schedule='fixed')
        avg_fixed += fixed.history

        frac = SMDS(d)
        frac.solve(15,debug=True,schedule='frac')
        avg_frac += frac.history

        exp = SMDS(d)
        exp.solve(15,debug=True,schedule='exp')
        avg_exp += exp.history

        sqrt = SMDS(d)
        sqrt.solve(15,debug=True,schedule='sqrt')
        avg_sqrt += sqrt.history

    avg_fixed = avg_fixed/n
    avg_frac = avg_frac/n
    avg_exp = avg_exp/n
    avg_sqrt = avg_sqrt/n

    x = np.arange(len(avg_fixed))
    plt.plot(x,avg_fixed,label="Fixed: 0.01")
    plt.plot(x,avg_exp,label="e^-bt")
    plt.plot(x,avg_frac,label="a/(b+t)")
    plt.plot(x,avg_sqrt,label="a/sqrt(b+t)")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Stress")
    plt.suptitle("Stress over iterations for different step sizes")
    plt.savefig('figs/30iterations-test.png')

    with open('30iterations-test.pkl', 'wb') as myfile:
        pickle.dump([avg_fixed,avg_frac,avg_exp,avg_sqrt], myfile)

def save_animation():
    G = subdivide_graph(gt.load_graph('graphs/cube.xml'),20)
    d = get_distance_matrix(G,verbose=False)

    n = 500

    stress = np.zeros(n)

    for _ in range(30):

        # frac = SMDS(d)
        # frac.solve(100,debug=True,schedule='frac')
        frac = SMDS(d)
        X = frac.solve(n,debug=True)

        name = 'new_outputs/cube_animation'
        output_sphere(G,X,name+'0.dot')


def distortion_plot():
    Vs = [i for i in range(20,100,5)]
    stress = np.zeros(len(Vs))
    for v in range(len(Vs)):
        G = subdivide_graph(gt.load_graph('graphs/cube.xml'),Vs[v])
        d = get_distance_matrix(G,verbose=False)

        for _ in range(30):

            frac = SMDS(d)
            frac.solve(100,debug=True,schedule='frac')
            stress[v] += frac.calc_distortion()
            name = 'new_outputs/cube_animation'
            for i in range(len(frac.pos_history)):
                output_sphere(G,frac.pos_history[i],name+str(i)+'.dot')

    stress /= 30
    plt.suptitle("Average distortion for increasing values of |V|")
    plt.xlabel("|V|")
    plt.ylabel("distortion")
    plt.plot(np.arange(len(stress)), stress)
    plt.savefig('figs/distortion_april')

def sphere_dist(xi,xj):
    sin, cos = np.sin, np.cos
    p1,l1 = xi
    p2,l2 = xj
    return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))

def dist_matrix(X,norm=np.linalg.norm):
    n = len(X)

    d = np.zeros( (n,n) )

    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = norm(X[i],X[j])

    return d

def compare_plot(n=2):
    Vs = [i for i in range(20,50,5)]
    m = len(Vs)
    stress = np.zeros( (m,n) )
    classic_stress = np.zeros( (m,n) )
    for v in range(m):
        #G = subdivide_graph(gt.load_graph('graphs/cube.xml'),Vs[v])
        G = gt.Graph(directed=False)
        G.add_vertex(n=v)

        x1 = np.random.uniform(0,2*math.pi, (Vs[v],1) )
        x2 = np.random.uniform(0,math.pi, (Vs[v],1) )
        X = np.concatenate( (x1,x2), axis=1 )


        d = dist_matrix( X, sphere_dist )
        print(d.shape)

        for i in range(n):

            # frac = SMDS(d)
            # frac.solve(100,debug=True,schedule='frac')
            # stress[v][i] += frac.calc_distortion()

            classic = MDS(d,geometry='spherical')
            classic.solve(500,debug=True)
            classic_stress[v][i] += classic.calc_distortion()

            name = 'new_outputs/cube_animation'
            for i in range(len(frac.pos_history)):
                output_sphere(G,frac.pos_history[i],name+str(i)+'.dot')

    stress /= n
    classic_stress /= n
    plt.suptitle("Average distortion for increasing values of |V|")
    plt.xlabel("|V|")
    plt.ylabel("distortion")
    plt.plot(Vs, stress.mean(axis=1),label='SGD')
    plt.plot(Vs,classic_stress.mean(axis=1),label='Classic')
    #plt.savefig('figs/distortion_april')
    plt.legend()
    plt.show()

def apsp(G):
    return np.array( [v for v in gt.shortest_distance(G)] ,dtype=float)

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )



def subdivide():
    G = gt.load_graph('graphs/cube.xml')
    d = apsp(G)
    print(d.max())

    y = np.linspace(0.45,2,100)
    data = np.zeros( (len(y), 2) )
    for i,a in enumerate(y):
        print(a)
        X = SGD(a*d,weighted=False).solve()
        data[i] = [a, distortion(X,a*d, lambda x1,x2: np.linalg.norm(x1-x2))]
    np.savetxt('data/cube_scale.txt',data,delimiter=',')

    return d.max()



def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d,metric=sphere_geo):
    dist = 0
    for i in range(len(X)):
        for j in range(i):
            dist += abs( metric(X[i],X[j]) - d[i][j]) / d[i][j]
    return dist/choose(len(X),2)
#save_animation()
#compare_plot(5)
n = 200
x1 = np.random.uniform(0, math.pi, (n,1) )
x2 = np.random.uniform(0,2*math.pi, (n,1) )
X = np.concatenate( (x1,x2), axis=1 )


#d = dist_matrix( X, sphere_dist )

# from sklearn.metrics import pairwise_distances
# d = pairwise_distances(X,metric='haversine')
#
# classic = MDS(d,geometry='spherical')
# classic.solve(500,debug=True)
#
# stochastic = SMDS(d)
# stochastic.solve(500,debug=True)
# print("Classic final distortion: {}".format(distortion(classic.X,d)))
# print("stochastic final distortion: {}".format(distortion(stochastic.X,d)))
#
# plt.plot(np.arange(len(classic.history)),classic.history)
# plt.plot(np.arange(len(stochastic.history)),stochastic.history)
# #plt.show()
# #X = np.ones((5,5))
# from autograd import grad
# import scipy.spatial.distance
# import autograd.numpy as np
#
#
# sin, cos, asin = np.sin, np.cos, np.arcsin
# sqrt = np.sqrt
# tol = np.ones( (X.shape[0], X.shape[0]) ) * 1e-13
#
# def haversine(X):
#     lat = X[:,0]
#     lng = X[:,1]
#     diff_lat = lat.reshape((n,1)) - lat
#     diff_lng = lng.reshape((n,1)) - lng
#     diff = sin(diff_lat/2)**2 + cos(lat.reshape((n,1)))*cos(lat) * sin(diff_lng/2)**2
#     Y =  2 * asin(sqrt(np.maximum(diff,tol)))
#     residual = (Y-d) ** 2
#     return residual.sum() / (n**2)
#
# print(haversine(X))
n = 15
diam = subdivide()

data = np.loadtxt('data/cube_scale.txt',delimiter=',')
X = data[:,0]
dist = data[:,1]
import pylab
pylab.plot(X,dist,label='Euclidean Distortion value')
#pylab.plot(np.ones(dist.shape[0])*(math.pi/diam),np.linspace(0,1,dist.shape[0]) ,'--',label='Proposed heuristic scale factor')
pylab.xlabel('scale factor')
pylab.ylabel('distortion')
pylab.legend()
#pylab.xlim(0,2)
pylab.ylim(0,0.8)
pylab.suptitle("Cube".format(diam, diam, round(math.pi/diam,5)))
pylab.show()

#save_animation()
