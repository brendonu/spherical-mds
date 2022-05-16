import ssgetpy
import numpy as np

result = ssgetpy.search(rowbounds=(None,2000),colbounds=(None,2000),limit=10000)

for mat in result:
    mat.download(format='MAT',destpath='/home/jacob/Desktop/spherical-mds/matrices')

import scipy.io as sio

import os
import pickle
import copy

path = 'matrices/'
graph_paths = os.listdir(path)
graph = graph_paths[0]

for graph in graph_paths:
    print(graph)
    try:
        mat = sio.loadmat(path+graph)['Problem'][0][0][1]
        mat = mat.toarray()
    except:
        try:
            mat = sio.loadmat(path+graph)['Problem'][0][0][2]
            mat = mat.toarray()
        except:
            try:
                mat = sio.loadmat(path+graph)['Problem'][0][0][3]
                mat = mat.toarray()
            except:
                print("Not expected format")
                continue
    #mat = mat.tolist()
    x1, x2 = mat.nonzero()
    x1 = x1.reshape( (len(x1),1))
    x2 = x2.reshape( (len(x1),1))
    y = np.not_equal(x1,x2)
    X = np.concatenate( (x1,x2), axis=1)
    X = np.delete(X, np.nonzero(~y),axis=0).astype(int)

    np.savetxt("txt_graphs/{}.txt".format(graph.split('.')[0]), X,fmt='%.d', delimiter=',' )
