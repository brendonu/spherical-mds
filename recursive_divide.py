import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS, all_pairs_shortest_path,output_sphere
from SGD_MDS import myMDS
from graph_functions import get_distance_matrix
import networkx as nx


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

H = gt.load_graph('graphs/cube.xml')

scores = []

for i in range(0,51,5):
    G = subdivide_graph(H.copy(),i)
    print(i)

    d = get_distance_matrix(G,verbose=False)

    spheres = []
    for i in range(5):
        Y = SMDS(d)
        Y.solve(200)
        spheres.append(Y.calc_distortion())
    spheres = np.array(spheres).mean()

    euclid = []
    for i in range(5):
        Z = myMDS(d)
        Z.solve()
        euclid.append(Z.calc_distortion())
    euclid = np.array(euclid).mean()

    scores.append({'i': i,
                   'sphere_score': spheres,
                   'euclid_score': euclid})


    #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
with open('sphere_scores_final.pkl', 'wb') as myfile:
    pickle.dump(scores, myfile)
