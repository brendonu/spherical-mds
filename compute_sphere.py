import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt


with open('sphere_scores_final.pkl', 'rb') as myfile:
    final = pickle.load(myfile)


spheres = np.zeros(len(final[0]))
euclids = np.zeros(len(final[0]))
sphere_median = np.zeros(len(final[0]))
euclid_median = np.zeros(len(final[0]))
sphere_violin = [None for i in range(len(final[0]))]
i = 0
for d in final[0]:
    spheres[i] = d['sphere_score']
    euclids[i] = d['euclid_score']
    sphere_median[i] = np.median(np.array(d['sphere_data']))
    euclid_median[i] = np.median(np.array(d['euclid_data']))
    sphere_violin[i] = np.array(d['sphere_data'])
    i += 1
x = np.array([(i*5)+8 for i in range(len(final[0]))])
print(sphere_median)
plt.suptitle("Spherical distortion on subdivided cubes")
plt.violinplot(sphere_violin,None,points=100,widths=1,showmeans=True,showextrema=True,showmedians=True)
plt.xticks(ticks=np.arange(len(x)),labels=x.astype('str'))
plt.xlabel("# of nodes")
plt.ylabel("Distortion")
plt.ylim(0,1)
plt.show()
plt.clf()

#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, sphere_median, label="Sphere Median")
plt.plot(x, euclid_median,label="Euclidean Median")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("# of nodes")
plt.ylabel("Average Distortion")
plt.xlim()
plt.ylim(0,1)
plt.suptitle("Subdivided cube (30 trials)")
plt.legend()

#G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

#plt.show()
plt.savefig("figs/test.png")
