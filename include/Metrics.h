#ifndef METRICS_H
#define METRICS_H

#include <string>
using namespace std;

class Metrics {

    public: 
        
        double mse(double y[], double y_hat[], int len);
        double mae(double y[], double y_pred[], int len);
        double r_squared(double sst, double ssr);

};

#endif