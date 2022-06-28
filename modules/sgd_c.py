import s_gd2
import graph_tool.all as gt
import numpy as np
import math
# G = gt.load_graph_from_csv("txt_graphs/1138_bus.txt",hashed=False)
#
# E = np.loadtxt('txt_graphs/1138_bus.txt',delimiter=',',dtype='int32')
#
# e1 = E[:,0]
# e2 = E[:,1]
#
#
# print('now')
# X = s_gd2.layout_convergent(e1,e2)
# print('end')
# #s_gd2.draw_svg(X, I, J, 'C5.svg')
# print(type(X))
# pos = G.new_vp('vector<float>')
# pos.set_2d_array(X.T)
# gt.graph_draw(G,pos=pos)


def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def distortion(X,d):
    epsilon = np.ones(d.shape)*1e-13
    N = len(X)
    ss = (X * X).sum(axis=1)
    diff = ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)
    diff = np.sqrt(np.maximum(diff,epsilon))
    stress = np.sum( np.divide(np.absolute(diff-d),d,out=np.zeros(d.shape),where= d!= 0) )

    return stress/math.comb(N,2)

import os
path = 'txt_graphs/'
graph_paths = os.listdir(path)

n = 10
data = np.empty( (len(graph_paths), n+1),dtype=np.dtype('U50') )
for j in range(len(graph_paths)):
    graph = graph_paths[j]
    print(graph)

    G = gt.load_graph_from_csv(path+graph,hashed=False)
    d = np.array( [v for v in gt.shortest_distance(G)] ,dtype=float)
    E = np.loadtxt(path+graph,delimiter=',',dtype='int32')

    data[j][0] = graph.split('.')[0]

    for i in range(n):
        try:
            X = s_gd2.layout_convergent(E[:,0],E[:,1])
        except:
            continue
        if i == 0:
            pos = G.new_vp('vector<float>')
            pos.set_2d_array(X.T)
            gt.graph_draw(G,pos=pos,output='sgd_drawings/{}.png'.format(graph.split('.')[0]))

        data[j][i+1] = distortion(X,d)

np.savetxt('data/sgd_suite.txt',data,delimiter=',',fmt='%s')
