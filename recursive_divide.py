import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS, all_pairs_shortest_path,output_sphere
from SGD_MDS import myMDS
from MDS_classic import MDS
from graph_functions import get_distance_matrix,subdivide_graph,subdivide_graph_recursive

import time




def compare_classic():
    H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
    count = 0

    scores = [[],[],[]]
    for graph in range(3):
        for i in range(0,50,5):
            G = subdivide_graph(H[graph].copy(),i)
            print(i)

            d = get_distance_matrix(G,verbose=False)

            spheres = []
            for i in range(5):
                Y = SMDS(d)
                Y.solve(200)
                spheres.append(Y.calc_distortion())
            output_sphere(G,Y.X,fname="outputs/final_run_" + str(count) + ".dot")

            spheres_mean = np.array(spheres).mean()

            euclid = []
            for i in range(5):
                Z = MDS(d,geometry='spherical')
                Z.solve(1000,debug=False)
                euclid.append(Z.calc_distortion())
            output_sphere(G,Z.X,fname="outputs/final_run_classic" + str(count) + ".dot")
            euclid_mean = np.array(euclid).mean()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_score': spheres_mean,
                           'classic_score': euclid_mean,
                           'stochastic_data': spheres,
                           'classic_data': euclid})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('compare_classic.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)


def compare_sphere_to_euclid():
    H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
    count = 0

    scores = [[],[],[]]
    for graph in range(3):
        for i in range(0,100,5):
            G = subdivide_graph(H[graph].copy(),i)
            print(i)
            print(G.num_vertices())

            d = get_distance_matrix(G,verbose=False)

            spheres = []
            for i in range(5):
                Y = SMDS(d)
                Y.solve(30)
                spheres.append(Y.calc_stress())

            spheres_mean = np.array(spheres).mean()

            euclid = []
            for i in range(5):
                Z = myMDS(d)
                Z.solve(30,debug=False)
                euclid.append(Z.calc_stress())
            euclid_mean = np.array(euclid).mean()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_score': spheres_mean,
                           'classic_score': euclid_mean,
                           'stochastic_data': spheres,
                           'classic_data': euclid,
                           'sphere_out': Y.X,
                           'euclid_out': Z.X})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/compare_sphere_to_euclid_update_linear_divide_stress.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)

if __name__ == "__main__":
    compare_sphere_to_euclid()
