import random
import numpy as np
from SGD_MDS_sphere import SMDS
from MDS_classic import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import time
import math

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )


def generate_spherical_data(n=100):
    x1 = np.random.uniform(0,math.pi, (n,1) )
    x2 = np.random.uniform(0,2*math.pi, (n,1) )
    return np.concatenate( (x1,x2), axis=1 )

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d):
    dist = 0
    for i in range(len(X)):
        for j in range(i):
            dist += abs( geodesic(X[i],X[j]) - d[i][j]) / d[i][j]
    return dist/choose(len(X),2)

def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(geodesic(X[i],X[j])-d[i][j],2) / pow(d[i][j],2)
    return stress / pow(len(X),2)


def compare(sizes = (20,100,10), iter=5):
    i,j,k = sizes
    num_data = [num for num in range(i,j,k)]

    data_classic = np.zeros( (iter*len(num_data), 3) )
    data_stoch = np.zeros( (iter*len(num_data), 3) )

    for size in range(len(num_data)):
        X = generate_spherical_data(n=num_data[size])
        d = pairwise_distances(X,metric='haversine')

        for a in range(iter):

            start_classic = time.perf_counter()
            X_c = MDS(d,geometry='spherical').solve(2000)
            end_classic = time.perf_counter()

            start_stoch = time.perf_counter()
            X_s = SMDS(d).solve(2000)
            end_stoch = time.perf_counter()

            data_classic[size*iter+a] = np.array([end_classic-start_classic, distortion(X_c,d),stress(X_c,d)])
            data_stoch[size*iter+a] = np.array([end_stoch-start_stoch, distortion(X_s,d),stress(X_s,d)])

    header_txt = "Starting from {}, size increases by {} every {} rows. Columns are time, distortion, stress".format(i,k,iter)
    np.savetxt('data/classic_results.csv',data_classic,delimiter=',',header=header_txt)
    np.savetxt('data/stochastic_results.csv',data_stoch,delimiter=',',header=header_txt)

compare(sizes=(20,500,10),iter=5)
