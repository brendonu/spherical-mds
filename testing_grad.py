import graph_tool.all as gt
import numpy as np

from SGD_MDS_sphere import SMDS
from SGD_MDS2 import SGD
from HMDS import HMDS
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import math

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
acosh, cosh, sinh = np.arccosh, np.cosh, np.sinh

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )
euclid_geo = lambda u,v: np.linalg.norm(u-v)

G = gt.load_graph_from_csv('txt_graphs/494_bus.txt')
gt.remove_parallel_edges(G)

interface = {
    'euclidean': SGD,
    'spherical': SMDS,
    'hyperbolic': HMDS
}

def apsp(G):
    return np.array( [v for v in gt.shortest_distance(G)] ,dtype=float)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d,metric=euclid_geo):
    dist = 0
    for i in range(len(X)):
        for j in range(i):
            dist += abs( metric(X[i],X[j]) - d[i][j]) / d[i][j]
    return dist/choose(len(X),2)

def compare(n=5):
    import os
    import pickle
    import copy

    path = 'txt_graphs/'
    graph_paths = os.listdir(path)

    for geom in ['euclidean', 'spherical', 'hyperbolic']:
        data = np.empty( (len(graph_paths), n+1),dtype=np.dtype('U50') )
        metric = euclid_geo if geom == 'euclidean' else sphere_geo if geom == 'spherical' else hyper_geo
        MDS = interface[geom]
        print(geom)

        for i in range(len(graph_paths)):
            graph = graph_paths[i]
            print(graph)
            G = gt.load_graph_from_csv(path+graph)
            gt.remove_parallel_edges(G)
            d = np.array( [v for v in gt.shortest_distance(G)] ,dtype=float)
            data[i,0] = graph.split('.')[0]

            for a in range(n):

                X = MDS(d).solve()

                data[i,a+1] = distortion(X,d,metric)

        np.savetxt("data/test.csv",data,delimiter=',',fmt='%s')

#compare(n=5)

G = gt.load_graph_from_csv('txt_graphs/1138_bus.txt')
d = apsp(G)

X = interface['hyperbolic'](d).solve(debug=True)
