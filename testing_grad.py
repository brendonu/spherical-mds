from autograd import grad
import autograd.numpy as np
import math


acosh, cosh, sinh, sqrt = np.arccosh, np.cosh, np.sinh, np.sqrt


from HMDS import HMDS
geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )

def generate_spherical_data(n=100):
    x1 = np.random.uniform(0,math.pi, (n,1) )
    x2 = np.random.uniform(0,math.pi, (n,1) )
    return np.concatenate( (x1,x2), axis=1 )

def generate_unit_circle(n=100):
    return np.random.uniform(0,1, (n,2) )

def dist_matrix(X,norm=np.linalg.norm):
    n = len(X)

    d = np.zeros( (n,n) )

    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = norm(X[i],X[j])

    return d

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d):
    dist = 0
    for i in range(len(X)):
        for j in range(i):
            dist += abs( hyper_geo(X[i],X[j]) - d[i][j]) / d[i][j]
    return dist/choose(len(X),2)

X = generate_spherical_data()
d = dist_matrix(X,norm=hyper_geo)
print(np.max(d))
# X = generate_unit_circle()
# d = dist_matrix(X,norm=hyper_geo)
# print(np.max(d))

Y = HMDS(d)
Y.solve(1000)
print(distortion(Y.X,d))
