import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt


with open('data/compare_sphere_to_euclid_update_linear_stress.pkl', 'rb') as myfile:
    final = pickle.load(myfile)


with open('compare_classic.pkl', 'rb') as myfile:
    standard = pickle.load(myfile)


spheres = np.zeros(len(final[0]))
euclids = np.zeros(len(final[0]))
sphere_median = np.zeros(len(final[0]))
euclid_median = np.zeros(len(final[0]))
sphere_violin = [None for i in range(len(final[0]))]
i = 0

for d in final[0]:
    spheres[i] = d['stochastic_dist']
    euclids[i] = d['standard_dist']
    sphere_median[i] = np.median(np.array(d['stochastic_time']))
    euclid_median[i] = np.median(np.array(d['standard_time']))
    sphere_violin[i] = np.array(d['standard_dist_data'])
    i += 1
print(standard[0][0].keys())
# for i in range(len(standard[0])):
#     euclid_median[i] = np.nanmedian(np.array(standard[0][i]['classic_data']))
#     print(standard[0][i]['classic_data'])
x = np.array([(i*5)+8 for i in range(len(final[0]))])
print(sphere_median)
plt.suptitle("Spherical distortion on subdivided cubes with smart initilization")
plt.violinplot(sphere_violin,None,points=100,widths=1,showmeans=True,showextrema=True,showmedians=True)
plt.xticks(ticks=np.arange(len(x)),labels=x.astype('str'))
plt.xlabel("# of nodes")
plt.ylabel("Distortion")
plt.ylim(0,1)
plt.show()
plt.clf()

#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, sphere_median, label="Stochastic Median")
plt.plot(x, euclid_median,label="Standard Median")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("# of nodes")
plt.ylabel("Time (s)")
plt.xlim()
#plt.yscale('log')
#plt.ylim(0,1)
plt.suptitle("Spherical MDS standard GD vs SGD (time)")
plt.legend()

#G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

plt.show()
#plt.savefig("figs/updated_sphereveuclid.png")
