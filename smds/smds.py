import numpy as np
from .swig import sgd as cpp
#import graph_tool.all as gt

__all__ = [
    "sgd",
]

def S_MDS(d,num_iter,eps,init=None):

    if d.shape[0] != d.shape[1] or d.shape[0] < 1 or len(d.shape) != 2:
        raise TypeError("Matrix must be 2-dimensional symmetric")

    if init:
        X = init 
    else:
        X = np.random.uniform(-1,1,(d.shape[0],2),dtype=np.float64)

    cpp.sgd(X,num_iter,eps)        