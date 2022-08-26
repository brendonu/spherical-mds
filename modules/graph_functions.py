import graph_tool.all as gt
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from numba import jit
import math

def sphere_stress(X,d,r=1):
    w = d.copy()
    w[w!=0] = w[w!=0]**-2
    diff = haversine_distances(X)
    ss = np.multiply(w,np.square(r*diff-d))
    return np.sum(ss)/ (2*X.shape[0]**2)

def distortion(X,d,metric=lambda x1,x2: np.linalg.norm(x1-x2)):
    dist = 0
    for i in range(X.shape[0]):
        for j in range(i):
            dist += abs( metric(X[i],X[j])-d[i,j] ) / d[i,j]
    return dist/math.comb(X.shape[0],2)

def apsp(G,weights=None):
    d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
    return d

def get_edge_list(G):
    E = np.zeros( (G.num_edges(), 2) )
    for i,(u,v) in enumerate(G.iter_edges()):
        E[i] = [u,v]
    return E

#From tsNET implementation

def get_shortest_path_distance_matrix(g, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.

    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

    return X


# Return the distance matrix of g, with the specified metric.
def get_distance_matrix(g, verbose=True, weights=None):
    if verbose:
        print('[distance_matrix] Computing distance matrix')

    X = get_shortest_path_distance_matrix(g, weights=weights)

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    if verbose:
        print('[distance_matrix] Done!')

    return X


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def cross(v1, v2):
    return Vector(v1.y * v2.z - v1.z * v2.y,
                  v1.z * v2.x - v1.x * v2.z,
                  v1.x * v2.y - v1.y * v2.x)

def det(v1, v2, v3):
    return dot(v1, cross(v2, v3))

class Pair:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

# Returns True if the great circle segment determined by s
# straddles the great circle determined by l
def straddles(s, l):
    print(det(s.v1, l.v1, l.v2) * det(s.v2, l.v1, l.v2))
    return det(s.v1, l.v1, l.v2) * det(s.v2, l.v1, l.v2) < 0

# Returns True if the great circle segments determined by a and b
# cross each other
def intersects(a, b):
    return straddles(a, b) and straddles(b, a)

def lat_long_to_cart(v):
        phi = v[1]
        theta = v[0]
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(theta)
        return np.array([x,y,z])

# Test. Note that we don't need to normalize the vectors.
#
# import math
# def polar(long,lat):
#     return Vector(math.cos(long)*math.cos(lat), math.sin(long)*math.cos(lat), math.sin(lat))
#
# n0 = polar(0.7007052816356313,5.212651188707541)
# n1 = polar(3.07525337476615,3.1279802552648364)
# n2 = polar(1.0083447462212305,3.96551909587155)
# n3 = polar(1.846486454434095,2.9836936482532725)
#
#
#
# print(n0)
# L1 = Pair(n0,n1)
# L2 = Pair(n2,n3)
#
# print(intersects(L1,L2))
# deg = math.pi/180
# print(intersects(Pair(polar(0,0), polar(50*deg,50*deg)),
#  Pair(polar(26*deg,25*deg), polar(80*deg,-30*deg))))
#
# print(-0.0<0)


def count_intersection(edges,X):
    edges = [(int(n1),int(n2)) for (n1,n2) in edges]
    print(edges)

    geodesics = list()

    count = 0
    for edge in edges:
        g1,g2 = edge
        p1 = polar(X[g1][0]%2*math.pi,X[g1][1]%(math.pi))
        p2 = polar(X[g2][0]%2*math.pi,X[g2][1]%(math.pi))
        geodesics.append((Pair(p1,p2),[g1,g2]))


    for i in range(len(geodesics)):
        for j in range(i+1,len(geodesics)):
            line1,g1 = geodesics[i]
            line2,g2 = geodesics[j]
            if intersects(line1,line2) and (g1[0] not in g2 and g1[1] not in g2):
                print(g1)
                print(g2)
                print()
                count += 1

    return count

def subdivide_graph(G,n):
    count = 0
    while count < n:
        E = list(G.edges())
        index = [i for i in range(len(E))]
        #random.shuffle(index)
        for i in index:
            U = G.add_vertex()
            G.add_edge(E[i].source(),U)
            G.add_edge(U,E[i].target())

            G.remove_edge(E[i])

            count += 1

            if count > n:
                break
    return G

def subdivide_graph_recursive(G,n):
    count = 0
    if n == 0:
        return G
    E = list(G.edges())
    index = [i for i in range(len(E))]
    for i in index:
        U = G.add_vertex()
        G.add_edge(E[i].source(),U)
        G.add_edge(U,E[i].target())

        G.remove_edge(E[i])

    return subdivide_graph_recursive(G.copy(),n-1)
