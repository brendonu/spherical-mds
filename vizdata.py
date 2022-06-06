import numpy as np
import matplotlib.pyplot as plt
import pylab
from experiments import exp_graphs

def polytopes():
    data = np.loadtxt('data/cube_polytopes.txt',delimiter=',')
    x = np.arange(data.shape[0])
    pylab.plot(x,data[:,0],label='euclid')
    pylab.plot(x,data[:,1],label='sphere')
    pylab.plot(x,data[:,2],label='hyperbolic')
    pylab.legend()
    pylab.show()


def geo_distortions():
    data = np.loadtxt('data/euclid_distortions.txt',delimiter=',')[10:25]
    s_data = np.loadtxt('data/spherical_distortions.txt',delimiter=',')[10:25]
    print(data)
    _,graphs = exp_graphs()
    graphs = [g[:8] for g in graphs]
    x = np.arange(data.shape[0])*3

    fig, ax = pylab.subplots()

    ax.set_xticks( x, labels=graphs[10:25] )
    pylab.bar(x-0.5,data[:,1],label="Euclidean")
    pylab.bar(x+0.5,s_data[:,1],label="Spherical")

    pylab.suptitle("Distortion comparison across graphs")
    pylab.ylabel('Distortion')
    pylab.legend()
    pylab.show()


geo_distortions()
