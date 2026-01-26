#include "../include/LinearAlgebra.h"

LinearAlgebra::LinearAlgebra(DataFrame& df) : data(df) {}

double LinearAlgebra::mean(double arr[]) {
    double arr_sum = kahan_sum(arr, data.n);
    return arr_sum / data.n;
}

double LinearAlgebra::squared_sum(double arr[], double mean) {

    double var = 0.0;

    for (int i = 0; i < data.n; i++) {
        double n_diff = arr[i] - mean;
        var += (n_diff * n_diff);
    }

    return var;
}

double LinearAlgebra::covariance(double x[], double y[], double x_mean, double y_mean) {

    double sum_x_y = 0.0;

    for (int i = 0; i < data.n; i++) {
        double x_diff = x[i] - x_mean;
        double y_diff = y[i] - y_mean;
        sum_x_y += x_diff * y_diff;
    }

    return sum_x_y;
}
