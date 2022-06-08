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
    data = np.loadtxt('data/euclid_distortions.txt',delimiter=',')[:10]
    s_data = np.loadtxt('data/spherical_distortions.txt',delimiter=',')[:10]
    h_data = np.loadtxt('data/hyperbolic_distortions1.txt',delimiter=',')[:10]
    print(data)
    _,graphs = exp_graphs()
    graphs = [g[:8] for g in graphs]
    x = np.arange(data.shape[0])*4

    fig, ax = pylab.subplots()

    e = data.mean(axis=1)
    s = s_data.mean(axis=1)

    ax.set_yticks( x, labels=graphs[:10] )
    pylab.barh(x-0.8,data[:,1],label="Euclidean")
    pylab.barh(x,s_data[:,1],label="Spherical")
    pylab.barh(x+0.8,h_data[:,1],label="hyperbolic")

    ax.invert_yaxis()

    pylab.suptitle("Distortion comparison across graphs")
    pylab.xlabel('Distortion')
    pylab.legend()
    pylab.show()


geo_distortions()
