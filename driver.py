import numpy as np
import graph_tool.all as gt
from graph_functions import apsp
from SGD_MDS_sphere import SMDS

G = gt.load_graph("graphs/cube.xml")
d = apsp(G)
X = SMDS(d).solve()
print(X)
