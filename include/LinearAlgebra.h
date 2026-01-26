#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include "DataFrame.h"

class LinearAlgebra {

private:
    DataFrame& data;
    double kahan_sum(double arr[], int len);

public:
    LinearAlgebra(DataFrame& df);

    double mean(double arr[]);
    double squared_sum(double arr[], double mean);
    double covariance(double x[], double y[], double x_mean, double y_mean);
};

#endif
