import numpy as np
import pylab


classic = np.loadtxt("data/update/classic_time.txt")
stoch = np.loadtxt("data/update/stoch_time.txt")

y1 = stoch.mean(axis=1)
y2 = classic.mean(axis=1)
x = [i for i in range(200,2001,200)]

pylab.plot(x,y1)
pylab.plot(x,y2)
pylab.show()
