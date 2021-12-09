import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt


with open('compare_classic.pkl', 'rb') as myfile:
    final = pickle.load(myfile)


spheres = np.zeros(len(final[0]))
euclids = np.zeros(len(final[0]))
data = []
print(final[0][0]['classic_data'])

for i in range(len(final[0])):
    data.append(np.array(final[0][i]['classic_data']))
    spheres[i] = final[0][i]['stochastic_score']
    euclids[i] = final[0][i]['classic_score']
    spheres[i] = np.median(np.array(final[0][i]['stochastic_data']))
    euclids[i] = np.median(np.array(final[0][i]['classic_data']))
print(data)
x = np.array([(i*5)+8 for i in range(len(final[0]))])

plt.violinplot(data, None, points=100, widths=1,
                      showmeans=True, showextrema=True, showmedians=True)
plt.show()
plt.clf()

#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, spheres, label="Median Stochastic")
plt.plot(x, euclids,label="Median Standard")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("# of nodes")
plt.ylabel("Average Distortion")
plt.xlim()
plt.ylim(0,1)
plt.suptitle("Subdivided cube")
plt.legend()

#G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

#plt.show()
plt.savefig("figs/subdivided_cube1.png")
