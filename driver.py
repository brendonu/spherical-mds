import numpy as np
import math
import graph_tool.all as gt
from graph_functions import apsp, sphere_stress
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS
from SGD_hyperbolic import HMDS
from SGD_MDS2 import SGD
from sklearn.metrics.pairwise import haversine_distances
from graph_functions import subdivide_graph_recursive


from numba import jit


sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
acosh, cosh, sinh = np.arccosh, np.cosh, np.sinh

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )
euclid_geo = lambda u,v: np.linalg.norm(u-v)

def stress(X,d,metric=euclid_geo):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(metric(X[i],X[j])-d[i][j],2) / pow(d[i][j],2)
    return stress / pow(len(X),2)

def sphere_drive():
    G = gt.load_graph("graphs/sierpinski3d.dot")
    #G = gt.load_graph_from_csv("txt_graphs/dwt_221.txt",hashed=False)
    print(G.num_vertices())

    import time

    start = time.perf_counter()
    d = apsp(G)
    X = SMDS(d,scale_heuristic=False).solve(epsilon=1e-9)
    end = time.perf_counter()
    print("Took {} seconds".format(end-start))
    print(stress(X,d))

    write_to_json(G,X)

def hyperbolic_drive():
    G = gt.load_graph('graphs/cube.xml')
    d = apsp(G)
    X = HMDS(d).solve()

data = np.loadtxt('graphs/mos.txt',delimiter=',',dtype='U100',skiprows=1)
labels = np.loadtxt('graphs/mos_labels.csv',delimiter=',',dtype='U100',skiprows=1)
disc_map = {}
for x in labels:
    disc_map[int(x[0])] = x[2]

print(data)

E = data[:,:3].astype(np.float64)
print(E)
G = gt.Graph(directed=False)

weights = G.new_edge_property('float')

G.add_edge_list(E,hashed=False,eprops=[weights])

G.remove_vertex(0)
for (e1,e2),w in zip(G.iter_edges(),weights):
    if disc_map[e1+1] == '13' and disc_map[e2+1] == '7':
        print("{},{}".format(e1,e2))
#
d = apsp(G,weights)

X = SGD(d,weighted=False).solve()
X = labels[:,3:].astype(np.float64)
L = labels[:,1]
mine = dict()
for l in L:
    if l not in mine:
        mine[l] = 0
    else:
        mine[l] += 1
        print('hello there')
X[:,1] *= -1

pos = G.new_vp('vector<float>')
pos.set_2d_array(X.T)

clr_map = G.new_vp('int')
halo = G.new_vp('string')
for v in G.iter_vertices():
    clr_map[v] = disc_map[v+1]
    if v == 93 or v == 98:
        halo[v] = 'red'
    else:
        halo[v] = 'black'



gt.graph_draw(G,pos=pos,vertex_fill_color=clr_map,vertex_color=halo,output='mapofscience.png')

X = SMDS(d,scale_heuristic=True).solve(epsilon=1e-9)
write_to_json(G,X)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d,metric=euclid_geo):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += abs(metric(X[i],X[j])-d[i][j]) / d[i][j]
    return stress / choose(X.shape[0],2)

def knn_graph(D,k=2):
    G = gt.Graph(directed=False)
    G.add_vertex(n=D.shape[0])
    ind = np.argsort(D,axis=1)
    ind = ind[:,1:k+1]

    for v in G.iter_vertices():
        for i in range(k): G.add_edge(v,ind[v,i])
    return G

def drive():
    G = gt.load_graph("graphs/isocahedron.xml")
    G = subdivide_graph_recursive(G,3)
    d = apsp(G)
    #G = gt.load_graph_from_csv("txt_graphs/dwt_221.txt",hashed=False)

    # D = np.loadtxt('City_Distance dataset.csv',delimiter=',', dtype='U100')
    # labels = D[1:,0]
    # print(labels)
    # D = np.loadtxt('worldcities.csv', delimiter=',', dtype='U100',skiprows=1)
    # cities = D[:,1]
    # Y = np.zeros( (len(labels),2) )
    #
    # for i,city in enumerate(labels):
    #     ind = np.where( cities == city )
    #     if len(ind[0]) <= 1:
    #         Y[i] = D[ind[0],2:4]
    #     else: Y[i] = D[ind[0][0], 2:4]
    # Y = np.radians(Y)
    #
    # d = haversine_distances(Y)
    #
    # # D = D[:,1:]
    # # D = D[1:,:]
    # # print(D)
    # # D[D == ''] = 0
    # # D = D.astype(np.float64)
    # # d = D + D.T
    # # print(D)
    # # G = knn_graph(D,k=4)
    #
    # #d *= (math.pi/3900)
    #
    # import time
    #
    # start = time.perf_counter()
    # d = apsp(G)


    X = SMDS(d).solve(epsilon=1e-15)
    #end = time.perf_counter()
    #print("Took {} seconds".format(end-start))
    print(stress(X,d))

    # G = gt.Graph(directed=False)
    # G.add_vertex(n=labels.shape[0])
    # names = G.new_vertex_property('string')
    #for v in G.iter_vertices(): names[v] = labels[v]

    write_to_json(G,X,name_map=None)

def all_graphs():
    import os
    import copy
    import random

    path = 'txt_graphs/'
    graph_paths = os.listdir(path)
    #graph_paths = ['block_model300.dot']

    graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    #graph_paths = ['netscience','block_model_500']

    graph_paths = random.sample(graph_paths,300)

    stresses = np.empty( (len(graph_paths),2),dtype='U100' )
    print(len(graph_paths))
    for i,graph in enumerate(graph_paths):
        print(graph)
        G = gt.load_graph_from_csv(path+graph+'.txt')
        print(G.num_vertices())
        d = apsp(G)
        try:
            X = SMDS(d,scale_heuristic=True).solve(epsilon=1e-5)
        except:
            continue
        stresses[i] = [graph, sphere_stress(X,d,1)]
        write_to_json(G,X,fname='webapp/store_data/{}.json'.format(graph))


def subdivide():
    graphs = [gt.load_graph("graphs/cube.xml"), gt.load_graph("graphs/dodecahedron.xml"), gt.load_graph("graphs/isocahedron.xml")]

    names = ['cube', 'dodecahedron', 'isocahedron']
    depth = 7

    for i,G in enumerate(graphs):
        data = np.zeros( (depth, 3) )
        for lvl in range(depth):
            print(lvl)
            H = subdivide_graph_recursive(G,lvl+1)
            d = apsp(H)

            X_E = SGD(d).solve()
            X_S = SMDS(d).solve(epsilon=1e-10)
            X_H = HMDS(d).solve()

            data[lvl] = [distortion(X_E,d,euclid_geo), distortion(X_S,d,sphere_geo),distortion(X_H,d,hyper_geo)]
        np.savetxt('data/{}_polytopes.txt'.format(names[i]),data,delimiter=',')

def subdivide_save():
    graphs = [gt.load_graph("graphs/cube.xml"), gt.load_graph("graphs/dodecahedron.xml"), gt.load_graph("graphs/isocahedron.xml")]

    names = ['cube', 'dodecahedron', 'isocahedron']
    depth = 7

    for i,G in enumerate(graphs):
        for lvl in range(depth):
            print(lvl)
            H = subdivide_graph_recursive(G,lvl)
            d = apsp(H)

            X_S = SMDS(d).solve(epsilon=1e-13)

            write_to_json(H,X_S,fname='webapp/store_data/{}_{}.json'.format(names[i],lvl))



if __name__ == "__main__":
    drive()
