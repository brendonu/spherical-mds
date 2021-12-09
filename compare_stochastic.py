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

    frac = SMDS(d)
    frac.solve(15,debug=True,schedule='frac')
    name = 'new_outputs/cube_animation'
    for i in range(len(frac.pos_history)):
        output_sphere(G,frac.pos_history[i],name+str(i)+'.dot')

save_animation()
