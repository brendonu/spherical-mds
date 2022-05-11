import numpy as np
import matplotlib.pyplot as plt

data_c = np.loadtxt('data/classic_results.csv',skiprows=1,delimiter=',')
data_s = np.loadtxt('data/stochastic_results.csv',skiprows=1,delimiter=',')

iter = 2
trait = 1

x1 = data_c[:,trait]
print(x1)
time_classic = x1.reshape( (len(data_c)//iter, iter)).mean(axis=1)

x2 = data_s[:,trait]

time_stoch = x2.reshape( ( len(data_s)//iter, iter ) ).mean(axis=1)
print(time_stoch)

y = list( range(20,41,10) )
plt.plot(y,time_classic,label="Standard GD time")
plt.plot(y,time_stoch, label="SGD time")
plt.legend()
plt.show()
