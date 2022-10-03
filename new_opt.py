import enum
import numpy as np
import graph_tool.all as gt
from collections import namedtuple
from modules.graph_functions import apsp

import math
import random


Pair = namedtuple("Pair", "vi vj dij")


def sgd(Pairs,n):
    X = np.random.uniform(0,1,(n,2))

    for pair in Pairs:
        i,j = Pair.vi, Pair.vj
        mag = np.linalg.norm(X[i]-X[j])
        


def main():
    G = gt.load_graph_from_csv("graphs/grid17.txt")


    #Temporary method 
    d = apsp(G)
    max_dist = d.max()
    md = np.where(d == d.max())
    K = {vi for m in md for vi in m }

    K = random.sample(tuple(K),k=4)
    print(K)

    Pairs = list()
    stdist = math.pi/max_dist


    for k in K:
        dist = gt.shortest_distance(G,k)
        for i,dik in enumerate(dist):
            if i != k: Pairs.append(Pair(i,k,stdist*dik))
    for u,v in G.iter_edges():
        Pairs.append(Pair(u,v,stdist))
    

    random.shuffle(Pairs,G.num_vertices())

    sgd(Pairs)
        

    

main()