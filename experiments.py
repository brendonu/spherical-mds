import numpy as np
import graph_tool.all as gt
from graph_functions import apsp,sphere_stress, distortion
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS
from HMDS import HMDS
from SGD_MDS2 import SGD
import pylab
import s_gd2

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
acosh, cosh, sinh = np.arccosh, np.cosh, np.sinh

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )
euclid_geo = lambda u,v: np.linalg.norm(u-v)

def stress_curve():
    #G = gt.load_graph("graphs/can_96.dot")
    G = gt.load_graph_from_csv("exp_graphs/cube.txt",hashed=False)
    print(G.num_vertices())

    paths,graphs = exp_graphs()
    paths = paths

    cap = [0.05,0.1,0.15,0.2,0.3]
    for c in cap:
        stresses = np.zeros(60)
        for g in paths:
            G = gt.load_graph_from_csv(g,hashed=False)
            d = apsp(G)
            Xs,rs = SMDS(d,scale_heuristic=True).solve(num_iter=60,debug=True,cap=c,epsilon=1e-9,schedule='convergent')
            stresses += np.array([sphere_stress(X,d,r) for X,r in zip(Xs,rs)])
        stresses /= 10
        pylab.plot(np.arange(len(stresses)),stresses,label="Upper bound: {}".format(c))
#    write_to_json(G,Xs[-1])
    #print(rs)

    pylab.xlabel("Iteration")
    pylab.ylabel("Distortion")
    pylab.suptitle("Average stress over benchmark graphs")
    pylab.legend()
    pylab.savefig('figs/upperbound_full.png')

def learning_rate(num_iter):

    rates = ['fixed', 'convergent', 'geometric', 'sqrt']
    paths, graphs = exp_graphs()

    data = np.empty( (0,60) )

    for i,(path,graph) in enumerate(zip(paths,graphs)):
        G = gt.load_graph_from_csv(path,hashed=False)
        print(G.num_vertices())
        d = apsp(G)

        stress_rate = np.zeros( (4,num_iter) )
        for j,rate in enumerate(rates):
            for k in range(5):
                Xs,rs = SMDS(d,scale_heuristic=True).solve(num_iter=num_iter,schedule=rate,debug=True,cap=0.1)
                stress_rate[j] += np.array([sphere_stress(X,d,r) for X,r in zip(Xs,rs)])
            stress_rate[j] /= 5
            pylab.plot(np.arange(stress_rate[j].shape[0]), stress_rate[j], label=rate)

        pylab.suptitle("Stress over iterations for different learning rates \n Graph: {}".format(graph))
        pylab.legend()
        pylab.yscale('log')
        pylab.savefig('figs/learning_rate/{}.png'.format(graph))
        pylab.clf()
        data = np.append(data,stress_rate,axis=0)

        np.savetxt('data/learning_rate_exp.txt',data,delimiter=',')


def euclid_compare(n=5):
    paths, graphs = exp_graphs()

    data = np.zeros( (len(paths),n) )

    for i ,(path,graph) in enumerate(zip(paths,graphs)):
        for j in range(n):
            G = gt.load_graph_from_csv(path,hashed=False)
            d = apsp(G)
            E = np.loadtxt(path,delimiter=',').astype(np.int32)
            u,v = E[:,0], E[:,1]

            X = s_gd2.layout_convergent(u,v)
            pos = G.new_vp('vector<float>')
            pos.set_2d_array(X.T)
            gt.graph_draw(G,pos=pos,output='sgd_drawings/{}.png'.format(graph))

            data[i,j] = distortion(X,d)
    np.savetxt('data/euclid_distortions.txt',data,delimiter=',')

def spherical_compare(n=5):
    paths, graphs = exp_graphs()

    data = np.zeros( (len(paths),n) )

    for i ,(path,graph) in enumerate(zip(paths,graphs)):
        for j in range(n):
            G = gt.load_graph_from_csv(path,hashed=False)
            d = apsp(G)

            X = SMDS(d,scale_heuristic=True).solve(epsilon=1e-9)
            write_to_json(G,X,fname='webapp/exp_drawings/{}.js'.format(graph))

            data[i,j] = distortion(X,d,sphere_geo)
    np.savetxt('data/spherical_distortions.txt',data,delimiter=',')


def exp_graphs():

    import os

    path = 'exp_graphs/'
    graph_paths = os.listdir(path)

    Gs = np.array([gt.load_graph_from_csv(path+graph,hashed=False).num_vertices() for graph in graph_paths])
    ind = np.argsort(Gs)
    graph_paths = [graph_paths[i] for i in ind]

    graph_paths_fmt = [x for x in map(lambda s: s.split('.')[0], graph_paths) ]
    return [path+graph for graph in graph_paths], graph_paths_fmt

def mnist():
    Y = np.loadtxt('mnist/mnist2500_X.txt')
    labels = np.loadtxt('mnist/mnist2500_labels.txt').astype(np.int64)

    Y = Y[:1200,:]
    labels = labels[:1200]

    from sklearn.metrics import pairwise_distances
    d = pairwise_distances(Y)
    X = SMDS(d,scale_heuristic=True).solve(epsilon=1e-9)

    G = gt.Graph(directed=False)
    G.add_vertex(n=labels.shape[0])
    names = G.new_vertex_property('string')
    for v in G.iter_vertices(): names[v] = labels[v]

    write_to_json(G,X,name_map=names,classes = list(labels))

def scale_curve(n=5):
    G = gt.price_network(20,directed=False)
    d = apsp(G)
    print(d.max())

    y = np.linspace(0.01,5,100)
    data_E = np.zeros( (len(y), 2) )
    data_S = np.zeros( (len(y), 2) )
    data_H = np.zeros( (len(y), 2) )

    for i,a in enumerate(y):
        print(a)
        e_dist,s_dist,h_dist = 0,0,0
        for _ in range(n):
            X_E = SGD(a*d,weighted=False).solve()
            X_S = SMDS(a*d,scale_heuristic=False).solve(epsilon=1e-9)
            X_H = HMDS(a*d).solve()

            e_dist += distortion(X_E,a*d, lambda x1,x2: np.linalg.norm(x1-x2))
            s_dist += distortion(X_S,a*d, sphere_geo)
            h_dist += distortion(X_H,a*d, hyper_geo)

        e_dist /= n
        s_dist /= n
        h_dist /= n

        data_E[i] = [a, e_dist]
        data_S[i] = [a, s_dist]
        data_H[i] = [a, h_dist]

    pylab.suptitle("Scalar factor of tree graph")
    pylab.plot(data_E[:,0],data_E[:,1],label="Euclidean MDS")
    pylab.plot(data_S[:,0],data_S[:,1],label="Spherical MDS")
    pylab.plot(data_H[:,0],data_H[:,1],label="Hyperbolic MDS")
    pylab.xlabel('scale factor')
    pylab.ylabel('distortion')
    pylab.legend()
    pylab.ylim(0,1)
    pylab.show()

    return d.max()



if __name__ == "__main__":
    print(exp_graphs())
