import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS, all_pairs_shortest_path,output_sphere
from SGD_MDS import myMDS
from SGD_hyperbolic import HMDS
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

            stochastics = []
            for i in range(5):
                Y = SMDS(d)
                Y.solve(200)
                stochastics.append(Y.calc_distortion())
            output_sphere(G,Y.X,fname="outputs/final_run_" + str(count) + ".dot")

            stochastics_mean = np.array(stochastics).mean()

            euclid = []
            for i in range(5):
                Z = MDS(d,geometry='spherical')
                Z.solve(1000,debug=False)
                euclid.append(Z.calc_distortion())
            output_sphere(G,Z.X,fname="outputs/final_run_classic" + str(count) + ".dot")
            euclid_mean = np.array(euclid).mean()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_score': stochastics_mean,
                           'classic_score': euclid_mean,
                           'stochastic_data': stochastics,
                           'classic_data': euclid})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('compare_classic.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)


def compare_sphere_to_euclid():
    H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
    count = 0

    scores = [[],[],[]]
    for graph in range(1):
        for i in range(0,50,5):
            G = subdivide_graph(H[graph].copy(),i)
            print(i)
            print(G.num_vertices())

            d = get_distance_matrix(G,verbose=False)

            stochastics_stress = []
            stochastics_dist = []
            stochastics_time = []
            for i in range(30):
                Y = SMDS(d)
                start = time.time()
                Y.solve(30,schedule='exp')
                stochastics_time.append(time.time()-start)
                stochastics_stress.append(Y.calc_stress())
                stochastics_dist.append(Y.calc_distortion())

            stochastics_stress_mean = np.array(stochastics_stress).mean()
            stochastics_dist_mean = np.array(stochastics_dist).mean()
            print("Mean for stochastic: " ,stochastics_dist_mean)


            standard_stress = []
            standard_dist = []
            standard_time = []
            for i in range(30):
                Z = MDS(d,geometry='spherical')
                start = time.time()
                Z.solve(100,debug=False)
                standard_time.append(time.time()-start)
                standard_stress.append(Z.calc_stress())
                standard_dist.append(Z.calc_distortion())
            standard_stress_mean = np.array(standard_stress).mean()
            standard_dist_mean = np.array(standard_dist).mean()
            print("Mean for standard: " ,standard_dist_mean)
            print()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_stress': stochastics_stress_mean,
                           'standard_stress': standard_stress_mean,
                           'stochastic_dist': stochastics_dist_mean,
                           'standard_dist': standard_dist_mean,
                           'stochastic_stress_data': stochastics_stress,
                           'stochastic_dist_data': stochastics_dist,
                           'standard_stress_data': standard_stress,
                           'standard_dist_data': standard_dist,
                           'stochastic_time': stochastics_time,
                           'standard_time': standard_time,
                           'stochastic_out': Y.X,
                           'standard_out': Z.X})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/compare_stochastic_to_standard4.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)


def compare_hyperbolic():
    H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
    count = 0

    scores = [[],[],[]]
    for graph in range(1):
        for i in range(0,50,5):
            H = nx.random_tree(i+10)
            G = gt.Graph(directed=False)
            G.add_vertex(n=len(H.nodes()))
            for e in H.edges():
                G.add_edge(e[0],e[1])
            print(i)
            print(G.num_vertices())

            d = get_distance_matrix(G,verbose=False)

            stochastics_stress = []
            stochastics_dist = []
            stochastics_time = []
            for i in range(30):
                Y = HMDS(d)
                start = time.time()
                Y.solve(30)
                stochastics_time.append(time.time()-start)
                stochastics_stress.append(Y.calc_stress())
                stochastics_dist.append(Y.calc_distortion())

            stochastics_stress_mean = np.array(stochastics_stress).mean()
            stochastics_dist_mean = np.array(stochastics_dist).mean()
            print("Mean for stochastic: " ,stochastics_dist_mean)


            standard_stress = []
            standard_dist = []
            standard_time = []
            for i in range(30):
                Z = MDS(d,geometry='hyperbolic')
                start = time.time()
                Z.solve(100,debug=False)
                standard_time.append(time.time()-start)
                standard_stress.append(Z.calc_stress())
                standard_dist.append(Z.calc_distortion())
            standard_stress_mean = np.array(standard_stress).mean()
            standard_dist_mean = np.array(standard_dist).mean()
            print("Mean for standard: " ,standard_dist_mean)
            print()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_stress': stochastics_stress_mean,
                           'standard_stress': standard_stress_mean,
                           'stochastic_dist': stochastics_dist_mean,
                           'standard_dist': standard_dist_mean,
                           'stochastic_stress_data': stochastics_stress,
                           'stochastic_dist_data': stochastics_dist,
                           'standard_stress_data': standard_stress,
                           'standard_dist_data': standard_dist,
                           'stochastic_time': stochastics_time,
                           'standard_time': standard_time,
                           'stochastic_out': Y.X,
                           'standard_out': Z.X})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/compare_hyperbolic_standard_stochastic.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)


def draw_graph(G,X,fname=None):
    # Convert layout to vertex property
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(X.T)
    if fname:
        gt.graph_draw(G, pos=pos,output='drawings/' + fname)
    else:
        # Show layout on the screen
        gt.graph_draw(G, pos=pos)


def test_3d():
    import math
    G,pos = gt.triangulation(np.random.rand(100,2)*4,type="delaunay")
    d = get_distance_matrix(G,verbose=False)

    from sklearn.manifold import MDS
    embedding = MDS(n_components=3,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(d)
    for i in range(len(X_transformed)):
        X_transformed[i] /= np.linalg.norm(X_transformed[i])
        print(np.linalg.norm(X_transformed[i]))
    draw_graph(G,X_transformed)

    H = [gt.load_graph('graphs/cube.xml'),gt.load_graph('graphs/dodecahedron.xml'),gt.load_graph('graphs/isocahedron.xml')]
    count = 0

    scores = [[],[],[]]
    for graph in range(1):
        for i in range(0,100,5):
            G = subdivide_graph(H[graph].copy(),i)
            print(i)
            print(G.num_vertices())

            d = get_distance_matrix(G,verbose=False)

            embedding = MDS(n_components=3,dissimilarity='precomputed')
            X_transformed = embedding.fit_transform(d)
            init = [None for i in range(len(d))]
            for i in range(len(X_transformed)):
                X_transformed[i] /= np.linalg.norm(X_transformed[i])
                x,y,z = X_transformed[i]
                theta = np.arctan(pow(x*x+y*y,0.5)/z)
                phi = np.arctan(y/x) if x > 0 else np.arctan(y/x)+math.pi if x < 0 else math.pi/2
                init[i] = [theta,phi]

            stochastics_stress = []
            stochastics_dist = []
            stochastics_time = []
            for i in range(30):
                Y = SMDS(d,init_pos=np.array(init))
                start = time.time()
                Y.solve(30,schedule='exp')
                stochastics_time.append(time.time()-start)
                stochastics_stress.append(Y.calc_stress())
                stochastics_dist.append(Y.calc_distortion())

            stochastics_stress_mean = np.array(stochastics_stress).mean()
            stochastics_dist_mean = np.array(stochastics_dist).mean()
            print("Mean for stochastic: " ,stochastics_dist_mean)


            # standard_stress = []
            # standard_dist = []
            # standard_time = []
            # for i in range(30):
            #     Y = SMDS(d)
            #     start = time.time()
            #     Y.solve(30,schedule='exp')
            #     standard_time.append(time.time()-start)
            #     standard_stress.append(Z.calc_stress())
            #     standard_dist.append(Z.calc_distortion())
            # standard_stress_mean = np.array(standard_stress).mean()
            # standard_dist_mean = np.array(standard_dist).mean()
            # print("Mean for standard: " ,standard_dist_mean)
            # print()

            count += 1
            scores[graph].append({'i': i,
                           'stochastic_stress': stochastics_stress_mean,
                           'stochastic_dist': stochastics_dist_mean,
                           'stochastic_stress_data': stochastics_stress,
                           'stochastic_dist_data': stochastics_dist,
                           'stochastic_time': stochastics_time,
                           'stochastic_out': Y.X})


        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/3dinit.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)

if __name__ == "__main__":
    #compare_sphere_to_euclid()
    compare_hyperbolic()
    #test_3d()
