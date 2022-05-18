import random
import numpy as np
from SGD_MDS_sphere import SMDS
from MDS_classic import MDS
from SGD_MDS2 import SGD
from HMDS import HMDS
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import time
import math

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
acosh, cosh, sinh = np.arccosh, np.cosh, np.sinh

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )
euclid_geo = lambda u,v: np.linalg.norm(u-v)

def generate_spherical_data(n=100):
    x1 = np.random.uniform(0,2*math.pi, (n,1) )
    x2 = np.random.uniform(0,math.pi, (n,1) )
    return np.concatenate( (x1,x2), axis=1 )

def generate_unit_circle(n=100):
    return np.random.uniform(0,0.5, (n,2) )

def dist_matrix(X,metric=euclid_geo):
    n = len(X)

    d = np.zeros( (n,n) )

    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = metric(X[i],X[j])

    return d

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

def stress(X,d,metric=euclid_geo):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(metric(X[i],X[j])-d[i][j],2) / pow(d[i][j],2)
    return stress / pow(len(X),2)


def compare(sizes = (20,100,10), iter=5):
    i,j,k = sizes
    num_data = [num for num in range(i,j,k)]

    for geom in ['spherical']:
        data = np.zeros( (iter*len(num_data), 3) )
        metric = euclid_geo if geom == 'euclidean' else sphere_geo if geom == 'spherical' else hyper_geo
        print(geom)

        for size in range(len(num_data)):
            if geom == 'spherical':
                X = generate_spherical_data(n=num_data[size])
            else:
                X = generate_unit_circle(n=num_data[size])

            d = dist_matrix(X,metric=sphere_geo)
            print(num_data[size])

            for a in range(iter):

                start = time.perf_counter()
                X_e = SGD(d).solve(1000)
                end = time.perf_counter()
                time_e = end-start

                start = time.perf_counter()
                X_s = SMDS(d).solve(num_iter=1000)
                end = time.perf_counter()
                time_s = end-start

                start = time.perf_counter()
                X_h = HMDS(d).solve(1000)
                end = time.perf_counter()
                time_h = end-start
                print("E: {}, S: {}, H: {}".format(distortion(X_e,d,euclid_geo), distortion(X_s,d,sphere_geo), distortion(X_h, d, hyper_geo)))

                data[size*iter+a] = np.array([distortion(X_e,d,euclid_geo),
                                              distortion(X_s,d,sphere_geo),
                                              distortion(X_h, d, hyper_geo)])



        header_txt = "Starting from {}, size increases by {} every {} rows. Columns are euclid, sphere, hyperbolic".format(i,k,iter)
        np.savetxt("data/{}_results.csv".format(geom),data,delimiter=',',header=header_txt)

compare(sizes=(100,1001,100),iter=3)
# #import autograd.numpy as np
# nlog, nabs, nconj = np.log, np.absolute, np.conj
#
# def poin_dist1(q,p):
#     top = ( nabs( 1- np.conj(q)*p ) + nabs( p-q ) )
#     bottom = nlog( nabs( 1- np.conj(q)*p ) - nabs( p-q ) )
#     return top-bottom
#
# def poin_dist2(q,p):
#     return nlog( ( nabs( 1- np.conj(q)*p ) + nabs( p-q ) ) / ( nabs( 1- np.conj(q)*p ) - nabs( p-q ) ) )
#
# def poin_dist3(q,p):
#     diff = abs(p-q)
#     first = abs( 1- q.conjugate()*p )
#     return math.log( (first+diff)/ (first-diff))
#
# import_mod = '''import numpy as np; import math; nlog, nabs, nconj = np.log, np.absolute, np.conj; import random;
# from __main__ import poin_dist1;
# from __main__ import poin_dist2;
# from __main__ import poin_dist3'''
#
# try1 = '''
# poin_dist1(random.random(),random.random())
# '''
#
# try2 = '''
# poin_dist2(random.random(),random.random())
# '''
#
# try3 = '''
# poin_dist3(random.random(),random.random())
# '''
#
# u = 0+0j
# v = 0.09+0.1j
# print(poin_dist1(u,v))
# print(poin_dist2(u,v))
# print(poin_dist3(u,v))
# import timeit
# print("Method 1 takes: {}".format( timeit.timeit(setup = import_mod,
#                      stmt = try1,
#                      number = 10000)/10000 ))
#
# print("Method 2 takes: {}".format( timeit.timeit(setup = import_mod,
#                      stmt = try2,
#                      number = 10000)/10000 ))
#
# print("Method 3 takes: {}".format( timeit.timeit(setup = import_mod,
#                      stmt = try3,
#                      number = 10000)/10000 ))
