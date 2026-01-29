#include "../include/Metrics.h"
 
double Metrics::mse(double y[], double y_hat[], int len){
    
    double sse = 0.0;

    for (int i = 0; i < len; i++) {
        sse += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }

    return sse / len; 
}

double Metrics::mae(double y[], double y_pred[], int len){

    double mae = 0.0;

    for (int i = 0; i < len; i++){

        if ((y[i] - y_pred[i]) < 0){
            mae += -(y[i] - y_pred[i]);
        } else{
            mae += (y[i] - y_pred[i]);
        }

    }
    return (mae / len);
}

double Metrics::r_squared(double sst, double ssr){

    double r_squared = 0.0;

    r_squared = 1 - (ssr / sst);
    
    return r_squared; 

}