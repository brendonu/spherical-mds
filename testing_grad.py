import graph_tool.all as gt
import numpy as np

G = gt.load_graph_from_csv('txt_graphs/494_bus.txt')
gt.remove_parallel_edges(G)
for edge in G.edges():
    print(edge)
