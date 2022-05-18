import numpy as np
import matplotlib.pyplot as plt

def read_and_parse(filename, iter=2, trait=1):
    data = np.loadtxt(filename,skiprows=1,delimiter=',')
    x = data[:,trait]
    return x.reshape( (len(data) // iter, iter ) ).mean(axis=1)

f_name = 'data/spherical_results.csv'
iter = 5
trait = 1


euclid = read_and_parse(f_name,iter=iter,trait=0)
sphere = read_and_parse(f_name,iter=iter,trait=1)
hyper = read_and_parse(f_name,iter=iter,trait=2)

x = [i*100+100 for i in range(len(euclid))]
plt.plot(x,euclid,label="Euclidean")
plt.plot(x,sphere,label="Spherical")
plt.plot(x,hyper,label="Hyperbolic")
plt.suptitle("Sampled from 2d spherical space")
#plt.yscale('log')
plt.ylabel("Distortion")
plt.xlabel("Number of data points")
plt.legend(title='Embedding space')
# plt.ylim(0,0.1)
plt.show()
