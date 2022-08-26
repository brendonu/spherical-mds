#distutils: language = c++

cdef extern from "sgd3.hpp":
    void layout(double X)

def aaa():
    layout(22)
    return 2