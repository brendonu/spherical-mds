#include <vector>
#include <cmath>
#include <stdexcept>

#include "sgd.hpp"

using std::vector;

void sgd_opt(double* X, vector<pair> &pairs, const vector<double> &steps)
{
    unsigned n_pairs = pairs.size();
    for (unsigned i_step=0; i_step<steps.size(); i_step ++)
    {
        const double step = steps[i_step];

        //Shuffle
        
        for (unsigned i_pair=0; i_pair<n_pairs; i_pair++)
        {
            const pair &t = pairs[i_pair];
            const int &i = t.i, &j = t.j;
            const double &w_ij = t.w;
            const double &d_ij = t.d;

            double mu = step * w_ij;
            if(mu>1)
                mu = 1;
            
            double dx = X[i*2] - X[j*2], dy = X[i*2+1]-X[j*2+1];
            double mag = sqrt(dx*dx + dy*dy);


            double r = (mu * (mag-d_ij)) / (2*mag);
            double r_x = r*dx;
            double r_y = r * dy;

            X[i*2] -= r_x;
            X[i*2+1] -= r_y;
            X[j*2] += r_x;
            X[j*2+1] += r_y;
        }
       
    }
}

vector<double> schedule(const vector<pair> &pairs, int num_iter, double eps){
    

    vector<double> steps;
    steps.reserve(num_iter);
    for (int i=0; i<num_iter; i++){
        steps.push_back(0.001);
    }
    return steps;
}

vector<pair> getpairs(int n, double* X)
{
    int nc2 = (n*(n-1))/2;
    vector<pair> pairs;
    pairs.reserve(nc2); 
}

void sgd(int n, double* X, int m, int num_iter, double eps)
{
    vector<pair> pairs = getpairs(n,D);
    vector<double> steps = schedule(pairs,num_iter,eps);
    sgd_opt(X,pairs,steps);
}