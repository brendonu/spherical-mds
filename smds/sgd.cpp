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
            
        }
       
    }
}