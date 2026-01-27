#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "DataFrame.h"
#include "LinearAlgebra.h"
#include "Metrics.h"
#include <iostream>
using namespace std;

struct Predictions {
    double* y_hat;
    int n;

    Predictions(int size){
        n = size;
        y_hat = new double[n];        
    }

    ~Predictions() {
        delete[] y_hat;
        y_hat = nullptr;
    }
};

struct RegressionResults {
    double beta_0;
    double beta_1;
    double x_mean;
    double y_mean;
    double ssx;
    double ssy;
    double sse;
    double sum_x_y;
    double mse;
    double mae;
    double r_squared;
};

class LinearRegression {

private:
    DataFrame& df;
    LinearAlgebra la;
    Metrics mt;

public:
    LinearRegression(DataFrame& data);

    RegressionResults train(Predictions& preds);
    void predict(Predictions& preds);

    void preview_predictions(const Predictions& preds, int n = 5);

    // your existing functions (kept private to user)
    void predict_y(double x[], double y_hat[], int len, double beta0, double beta1);
    void beta1(const DataFrame& df);
    double beta0(double x_mean, double y_mean, double beta1);
    double sse(double y[], double y_hat[], int len);
};

#endif
