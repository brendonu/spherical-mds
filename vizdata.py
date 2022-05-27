import numpy as np
import matplotlib.pyplot as plt
import pylab

def read_and_parse(filename, iter=2, trait=1):
    data = np.loadtxt(filename,skiprows=1,delimiter=',')
    x = data[:,trait]
    return x.reshape( (len(data) // iter, iter ) ).mean(axis=1)

# f_name = 'data/spherical_results.csv'
# iter = 3
# trait = 1
#
#
# euclid = read_and_parse(f_name,iter=iter,trait=0)
# sphere = read_and_parse(f_name,iter=iter,trait=1)
# hyper = read_and_parse(f_name,iter=iter,trait=2)
#
# x = [i*100+100 for i in range(len(euclid))]
# plt.plot(x,euclid,label="Euclidean")
# plt.plot(x,sphere,label="Spherical")
# plt.plot(x,hyper,label="Hyperbolic")
# plt.suptitle("Sampled from 2d spherical space")
# #plt.yscale('log')
# plt.ylabel("Distortion")
# plt.xlabel("Number of data points")
# plt.legend(title='Embedding space')
# # plt.ylim(0,0.1)
# plt.show()

f_name = 'data/classic_results1.csv'
iter = 5
trait = 2

time_classic0 = read_and_parse(f_name,iter=iter,trait=trait)
time_stochastic0 = read_and_parse('data/stochastic_results1.csv',iter=iter,trait=trait)

time_classic1 = read_and_parse('data/classic_results2.csv',iter=1,trait=trait)
time_stochastic1 = read_and_parse('data/stochastic_results2.csv',iter=1,trait=trait)


x = [i*100 + 100 for i in range(len(time_classic0))] + [i*200 + 1000 for i in range(len(time_classic1))]
pylab.plot(x[1:],list(time_classic0)[1:]+list(time_classic1), '-o' ,label='GD')
pylab.plot(x[1:],list(time_stochastic0)[1:] + list(time_stochastic1), '-o',label='SGD')
#pylab.yscale('log')
pylab.ylabel("Stress")
pylab.xlabel("Number of data points")
pylab.ylim(-0.01,0.8)
pylab.legend()
pylab.suptitle("Average stress")
pylab.show()
