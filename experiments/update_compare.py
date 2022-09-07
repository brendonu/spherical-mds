import numpy as np
import pylab
import time
from sklearn.metrics.pairwise import haversine_distances

from modules.SGD_MDS_sphere import SMDS
from modules.MDS_classic import MDS
from modules.graph_functions import sphere_stress

def generate_spherical_data(n=100):
    x1 = np.random.uniform(0,2*np.pi, (n,1) )
    x2 = np.random.uniform(0,np.pi, (n,1) )
    return np.concatenate( (x1,x2), axis=1 )

gen = lambda: np.zeros( (10,20) )

stochastic,stoch_count, classic, classic_count = gen(), gen(), gen(), gen()
stoch_time, classic_time = gen(),gen()
for i,n in enumerate([200,400,600,800,1000,1200,1400,1600,1800,2000]):
    print("{} out of 10 graphs.".format(i))
    for j in range(15):
        print("Iteration number {} out of 15".format(j))
        points = generate_spherical_data(n)
        d = haversine_distances(points)

        start = time.perf_counter()
        X,count = SMDS(d).solve(num_iter=400)
        stoch_time[i,j] = time.perf_counter() - start
        stochastic[i,j] = sphere_stress(X,d)
        stoch_count[i,j] = count

        start = time.perf_counter()
        X,count,hist = MDS(d).solve(num_iter=400)
        classic_time[i,j] = time.perf_counter() - start
        classic[i,j] = sphere_stress(X,d)
        classic_count[i,j] = count

        print("---------------------------")

np.savetxt("data/update/stochastic.txt",stochastic)
np.savetxt("data/update/stoch_count.txt",stoch_count)
np.savetxt("data/update/stoch_time.txt",stoch_time)
np.savetxt("data/update/classic.txt",classic)
np.savetxt("data/update/classic_count.txt",classic_count)
np.savetxt("data/update/classic_time.txt",classic_time)
