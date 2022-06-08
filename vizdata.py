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
    data = np.loadtxt('data/euclid_distortions.txt',delimiter=',')[35:]
    s_data = np.loadtxt('data/spherical_distortions.txt',delimiter=',')[35:]
    h_data = np.loadtxt('data/hyperbolic_distortions.txt',delimiter=',')[35:]
    print(data)
    _,graphs = exp_graphs()
    graphs = [g[:8] for g in graphs]
    x = np.arange(data.shape[0])*4

    fig, ax = pylab.subplots()

    # e = data.mean(axis=1)
    # s = s_data.mean(axis=1)

    ax.set_yticks( x, labels=graphs[35:] )
    pylab.barh(x-0.8,data[:,1],label="Euclidean")
    pylab.barh(x,s_data[:,1],label="Spherical")
    pylab.barh(x+0.8,h_data[:,1],label="hyperbolic")

    ax.invert_yaxis()

    pylab.suptitle("Distortion comparison across graphs")
    pylab.xlabel('Distortion')
    pylab.legend()
    pylab.show()

def lr_plot():
    data = np.loadtxt('data/learning_rate_exp.txt',delimiter=',')
    print(data.shape)
    lrs_fix = []
    for i in range(data.shape[0]):
        if i % 4 == 0:
            lrs_fix.append(data[i])

    lrs_fix = np.array(lrs_fix)
    print(lrs_fix.shape)
    pylab.plot(np.arange(lrs_fix.shape[1]), lrs_fix.mean(axis=0))
    pylab.show()

def distortion_table():
    #matplotlib.pyplot.table(cellText=None, cellColours=None, cellLoc='right',
    #colWidths=None, rowLabels=None, rowColours=None, rowLoc='left', colLabels=None,
    #colColours=None, colLoc='center', loc='bottom', bbox=None, edges='closed', **kwargs)
    e_data = np.delete(np.loadtxt('data/euclid_distortions.txt',delimiter=',').mean(axis=1), [21,22,24],axis=0)
    s_data = np.delete(np.loadtxt('data/spherical_distortions.txt',delimiter=',').mean(axis=1), [21,22,24],axis=0)
    h_data = np.delete(np.nanmean(np.loadtxt('data/hyperbolic_distortions2.txt',delimiter=','),axis=1), [21,22,24],axis=0)[:32]
    h_d_2 = np.loadtxt('data/hyperbolic_distortions.txt',delimiter=',').mean(axis=1)[35:]

    print(h_data.shape)
    print(h_d_2.shape)
    h_data = np.append(h_data,h_d_2,axis=0)


    _,graphs = exp_graphs()
    print(graphs)


    print(np.column_stack( (e_data,s_data,h_data)).shape)

    fig, ax = pylab.subplots()
    pylab.axis('off')
    pylab.table(cellText=np.round( np.column_stack( (e_data,s_data,h_data)) ,decimals=5),
        rowLabels=graphs, loc='center')
    pylab.show()



distortion_table()
