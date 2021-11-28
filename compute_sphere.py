import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt


with open('sphere_scores_final.pkl', 'rb') as myfile:
    final = pickle.load(myfile)


spheres = np.zeros(len(final))
euclids = np.zeros(len(final))

for i in range(len(final)):
    spheres[i] = final[i]['sphere_score']
    euclids[i] = final[i]['euclid_score']
x = np.array([(i*5)+8 for i in range(len(final))])


#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, spheres, label="Sphere")
plt.plot(x, euclids,label="Euclidean")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("# of nodes")
plt.ylabel("Average Distortion")
plt.xlim()
plt.ylim()
plt.suptitle("Subdivided cube")
plt.legend()

#G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

#plt.show()
plt.savefig("figs/subdivided_cube1.png")
