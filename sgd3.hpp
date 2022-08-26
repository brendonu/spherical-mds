#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <vector>
using std::vector;

void layout(double X);



struct term
{
    int i,j;
    double d,w;
    term(int i, int j, double d, double w) : i(i), j(j), d(d), w(w) {}
};

void sgd(double* X);


#endif