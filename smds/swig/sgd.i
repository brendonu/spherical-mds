%module sgd
%{
    #define SWIG_FILE_WITH_INIT
    #include "../sgd.hpp"
%}

%include "numpy.i"
%init %{
    import_array();
%}

// vertex positions
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2){(double* X, int n, int kd)}

// edge indices
%apply (int* IN_ARRAY1, int DIM1){(int* I, int len_I),
                                  (int* J, int len_J)}
%apply (double* IN_ARRAY1, int DIM1){(double* V, int len_V)}

// for direct MDS with weights given
%apply (double* IN_ARRAY1, int DIM1){(double* d, int len_d),
                                     (double* w, int len_w),
                                     (double* eta, int len_eta)}

#include "../sgd.hpp"

%rename (sgd) np_sgd;


%inline %{
    void np_sgd(double* X, int n, int kd, 
                int max_iter, double eps){
                    sgd(n,X,max_iter,eps);
                }
%}