import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS, all_pairs_shortest_path,output_sphere
from SGD_MDS import myMDS
from MDS_classic import MDS
from graph_functions import get_distance_matrix

import time


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

# H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
# count = 0
#
# scores = [[],[],[]]
# for graph in range(3):
#     for i in range(0,50,5):
#         G = subdivide_graph(H[graph].copy(),i)
#         print(i)
#
#         d = get_distance_matrix(G,verbose=False)
#
#         spheres = []
#         sphere_time = []
#         for i in range(5):
#             Y = SMDS(d)
#             start = time.time()
#             Y.solve(200)
#             end = time.time()
#
#             spheres.append(Y.calc_distortion())
#             sphere_time.append(end-start)
#         output_sphere(G,Y.X,fname="outputs/final_run_" + str(count) + ".dot")
#
#         spheres_mean = np.array(spheres).mean()
#
#         euclid = []
#         euclid_time = []
#         for i in range(5):
#             Z = MDS(d,geometry='spherical')
#             start = time.time()
#             Z.solve(1000,debug=False)
#             end = time.time()
#             euclid.append(Z.calc_distortion())
#             euclid_time.append(end-start)
#         output_sphere(G,Z.X,fname="outputs/final_run_classic" + str(count) + ".dot")
#         euclid_mean = np.array(euclid).mean()
#
#         count += 1
#         scores[graph].append({'i': i,
#                        'stochastic_score': spheres_mean,
#                        'classic_score': euclid_mean,
#                        'stochastic_data': spheres,
#                        'classic_data': euclid,
#                        'stochastic_time': sphere_time,
#                        'classic_time': euclid_time})
#
#         with open('compare_classic.pkl', 'wb') as myfile:
#             pickle.dump(scores, myfile)
#
#
#     #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
# with open('compare_classic.pkl', 'wb') as myfile:
#     pickle.dump(scores, myfile)
