#ifndef SGD_HPP
#define SGD_HPP

#include <vector> 
using std::vector;

///////////////////////
// visible to python //
///////////////////////
void sgd(int n, double* X, int m, int num_iter, double eps);



//////////////
// cpp only //
//////////////
struct pair
{
    int i,j;
    double d,w;
    pair(int i, int j, double d, double w) : i(i), j(j), d(d), w(w) {}
};

void sgd_opt(double* X, vector<pair> &pairs, const vector<double> &steps);

#endif