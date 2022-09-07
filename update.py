import graph_tool.all as gt 
import numpy as np

import modules.SGD_MDS2 as sgd
from modules.graph_functions import apsp


G = gt.load_graph_from_csv("graphs/494_bus.txt")
d = apsp(G)
n = G.num_vertices()

terms = np.array([ (int(i),int(j),d[i,j]) for j in range(n) for i in range(n) if i != j])
steps = sgd.schedule_convergent(d,30,0.1,200)

Y = np.random.uniform(0,1,(n,2))

X = sgd.sgd2(Y,terms,steps,1e-5)