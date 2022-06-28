from numba import jit
import math
import numpy as np

@jit(nopython=True,fastmath=True,cache=True)
def math_sin(X):
    s = 0
    sin = math.sin
    for i in range(X.shape[0]):
        s += sin(X[i])
    return s

@jit(nopython=True,fastmath=True,cache=True)
def np_sin(X):
    s,sin = 0,np.sin
    for i in range(X.shape[0]):
        s += sin(X[i])
    return s

import timeit
print(timeit.repeat("math_sin(np.random.random(10000))", "import numpy as np; import math; from numba import jit; from __main__ import math_sin",number=10000))
print(timeit.repeat("np_sin(np.random.random(10000))", "import numpy as np; import math; from numba import jit; from __main__ import np_sin",number=10000))
