import numpy as np
import graph_tool.all as gt
from graph_functions import apsp,sphere_stress
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS


def stress_curve():
    #G = gt.load_graph("graphs/can_96.dot")
    G = gt.load_graph_from_csv("txt_graphs/dwt_221.txt",hashed=False)
    print(G.num_vertices())


    d = apsp(G)
    Xs,rs = SMDS(d,scale_heuristic=False).solve(debug=True,cap=0.5)
    write_to_json(G,Xs[-1])
    print(rs)

    stresses = [stress(X,d,r) for X,r in zip(Xs,rs)]
    import pylab
    pylab.plot(np.arange(len(stresses)),stresses)
    pylab.show()

def learning_rate():
    G = gt.load_graph_from_csv("txt_graphs/can_96.txt",hashed=False)
    print(G.num_vertices())


    d = apsp(G)
    Xs,rs = SMDS(d,scale_heuristic=True).solve(schedule='convergent',debug=True,cap=0.5)
    write_to_json(G,Xs[-1])
    print(rs)

    stresses = [stress(X,d,r) for X,r in zip(Xs,rs)]
    compare = [sphere_stress(X,d,r) for X,r in zip(Xs,rs)]
    import pylab
    pylab.plot(np.arange(len(stresses)),stresses)
    pylab.plot(np.arange(len(stresses)),compare)

    pylab.show()


learning_rate()
