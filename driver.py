import numpy as np
import graph_tool.all as gt
from graph_functions import apsp
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS
from SGD_MDS2 import SGD

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(geodesic(X[i],X[j])-d[i][j],2) / pow(d[i][j],2)
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
