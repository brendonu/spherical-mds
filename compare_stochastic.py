import networkx as nx
import graph_tool.all as gt
import random
import pickle
import numpy as np
from SGD_MDS_sphere import SMDS, all_pairs_shortest_path,output_sphere
from SGD_MDS import myMDS
from MDS_classic import MDS
from graph_functions import *
import matplotlib.pyplot as plt
from recursive_divide import subdivide_graph

def sphere_dist(xi,xj):
    sin, cos = np.sin, np.cos
    l1, p1 = xi
    l2, p2 = xj
    return np.arccos(sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(l2-l1))

def get_stress(X,d):
    stress = 0
    for i in range(len(d)):
        for j in range(i):
            stress += ((d[i][j] - sphere_dist(X[i],X[j])) ** 2 ) / d[i][j] ** 2
    return stress

def generate_stress_plots():
    G = gt.load_graph('outputs/bad_example.dot')
    #G = gt.complete_graph(4)
    d = get_distance_matrix(G,verbose=False)

    n = 30

    avg_fixed = np.zeros(200)
    avg_frac = np.zeros(200)
    avg_exp = np.zeros(200)
    avg_sqrt = np.zeros(200)

    for i in range(n):
        fixed = SMDS(d)
        fixed.solve(15,debug=True,schedule='fixed')
        avg_fixed += fixed.history

        frac = SMDS(d)
        frac.solve(15,debug=True,schedule='frac')
        avg_frac += frac.history

        exp = SMDS(d)
        exp.solve(15,debug=True,schedule='exp')
        avg_exp += exp.history

        sqrt = SMDS(d)
        sqrt.solve(15,debug=True,schedule='sqrt')
        avg_sqrt += sqrt.history

    avg_fixed = avg_fixed/n
    avg_frac = avg_frac/n
    avg_exp = avg_exp/n
    avg_sqrt = avg_sqrt/n

    x = np.arange(len(avg_fixed))
    plt.plot(x,avg_fixed,label="Fixed: 0.01")
    plt.plot(x,avg_exp,label="e^-bt")
    plt.plot(x,avg_frac,label="a/(b+t)")
    plt.plot(x,avg_sqrt,label="a/sqrt(b+t)")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Stress")
    plt.suptitle("Stress over iterations for different step sizes")
    plt.savefig('figs/30iterations-test.png')

    with open('30iterations-test.pkl', 'wb') as myfile:
        pickle.dump([avg_fixed,avg_frac,avg_exp,avg_sqrt], myfile)

def save_animation():
    G = subdivide_graph(gt.load_graph('graphs/cube.xml'),20)
    d = get_distance_matrix(G,verbose=False)

    stress = np.zeros(51)

    for _ in range(30):

        frac = SMDS(d)
        frac.solve(100,debug=True,schedule='frac')

        name = 'new_outputs/cube_animation'
        for i in range(len(frac.pos_history)):
            output_sphere(G,frac.pos_history[i],name+str(i)+'.dot')
            stress[i] += get_stress(frac.pos_history[i],d)
    stress /= 30
    plt.suptitle("Average stress over iteration on SMDS")
    plt.xlabel("Iteration")
    plt.ylabel("Stress")
    plt.plot(np.arange(len(stress)), stress)
    plt.show()

def distortion_plot():
    Vs = [i for i in range(20,100,5)]
    stress = np.zeros(len(Vs))
    for v in range(len(Vs)):
        G = subdivide_graph(gt.load_graph('graphs/cube.xml'),Vs[v])
        d = get_distance_matrix(G,verbose=False)

        for _ in range(30):

            frac = SMDS(d)
            frac.solve(100,debug=True,schedule='frac')
            stress[v] += frac.calc_distortion()
            name = 'new_outputs/cube_animation'
            for i in range(len(frac.pos_history)):
                output_sphere(G,frac.pos_history[i],name+str(i)+'.dot')

    stress /= 30
    plt.suptitle("Average distortion for increasing values of |V|")
    plt.xlabel("|V|")
    plt.ylabel("distortion")
    plt.plot(np.arange(len(stress)), stress)
    plt.savefig('figs/distortion_april')


#save_animation()
distortion_plot()
