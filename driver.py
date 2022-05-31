import numpy as np
import graph_tool.all as gt
from graph_functions import apsp
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
geodesic = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(geodesic(X[i],X[j])-d[i][j],2) / pow(d[i][j],2)
    return stress / pow(len(X),2)

G = gt.load_graph("graphs/oscil.dot")
#G = gt.load_graph_from_csv("txt_graphs/dwt_221.txt",hashed=False)
print(G.num_vertices())

import time

start = time.perf_counter()
d = apsp(G)
X = SMDS(d,scale_heuristic=True).solve()
end = time.perf_counter()
print("Took {} seconds".format(end-start))
print(stress(X,d))

write_to_json(G,X)
