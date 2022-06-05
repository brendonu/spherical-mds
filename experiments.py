import numpy as np
import graph_tool.all as gt
from graph_functions import apsp,sphere_stress, distortion
from graph_io import write_to_json
from SGD_MDS_sphere import SMDS
import s_gd2

sin,cos, acos, sqrt = np.sin, np.cos, np.arccos, np.sqrt
acosh, cosh, sinh = np.arccosh, np.cosh, np.sinh

sphere_geo = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
hyper_geo = lambda u,v: acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )
euclid_geo = lambda u,v: np.linalg.norm(u-v)

def stress_curve():
    #G = gt.load_graph("graphs/can_96.dot")
    G = gt.load_graph_from_csv("txt_graphs/dwt_1005.txt",hashed=False)
    print(G.num_vertices())


    d = apsp(G)
    Xs,rs = SMDS(d,scale_heuristic=True).solve(debug=True,cap=0.5,epsilon=1e-9,schedule='convergent')
    write_to_json(G,Xs[-1])
    #print(rs)

    stresses = [sphere_stress(X,d,r) for X,r in zip(Xs,rs)]
    import pylab
    pylab.plot(np.arange(len(stresses)),stresses)
    pylab.show()

def learning_rate(num_iter):
    import pylab

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
                Xs,rs = SMDS(d,scale_heuristic=True).solve(num_iter=num_iter,schedule=rate,debug=True,cap=0.5)
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

if __name__ == "__main__":
    learning_rate(60)
