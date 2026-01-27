#include "../include/LinearRegression.h"
#include "../include/LinearAlgebra.h"
#include "../include/Metrics.h"

LinearRegression::LinearRegression(DataFrame& data)
    : df(data), la(data) {}


RegressionResults LinearRegression::train(Predictions& preds) {

    RegressionResults res;

    int len = df.n;

    res.x_mean = la.mean(df.x);
    res.y_mean = la.mean(df.y);

    res.ssx = la.squared_sum(df.x, res.x_mean);
    res.ssy = la.squared_sum(df.y, res.y_mean);

    res.sum_x_y = la.covariance(df.x, df.y, res.x_mean, res.y_mean);

    predict_y(df.x, preds.y_hat, len, res.beta_0, res.beta_1);

    res.sse = sse(df.y, preds.y_hat, len);
    res.mae = mt.mae(df.y, preds.y_hat, len);
    res.r_squared = mt.r_squared(res.ssy, res.sse);

    return res;
}



// y_hat prediction 
void LinearRegression::predict(Predictions& preds){
    predict_y(df.data, preds.y_hat, df.n, df.beta_hat[0], df.beta_hat[1]);
}



void LinearRegression::preview_predictions(const Predictions& preds, int n = 5){

    if (n > preds.n) {
        cout << "n cannot be greater than the length of the dataset" << endl;
        return;
    }

    for (int i = 0; i < n; i++ ){
        cout << preds.y_hat[i] << endl;
    }
}

void LinearRegression::beta1(const DataFrame& df) {

    int num_vars = df.x_len;

    // calculating the X^TX the hard way, will replace with cholesky soon
    double* XtX = new double[num_vars * num_vars];
    double* Xty = new double[num_vars];

    for (int i = 0; i < num_vars * num_vars; i++) {
        XtX[i] = 0.0;
    } 

    // X^tX
    for (int i = 0; i < num_vars; i++) {          // column i
        for (int j = 0; j < num_vars; j++) {      // column j
            double sum = 0.0;

            for (int k = 0; k < df.n; k++) {  // rows
                sum += df.x[k * num_vars + i] * df.x[k * num_vars + j];
            }

            XtX[i * num_vars + j] = sum;
        }
    }

    //Xty
    for (int i = 0; i < num_vars; i++) {
        double sum = 0.0;

        for (int k = 0; k < df.n; k++) {
            sum += df.x[k * num_vars + i] * df.y[k];
        }

        Xty[i] = sum;
    }

    // LU decomposition for matrix inverse
    for (int k = 0; k < num_vars; k++) {

        for (int i = k + 1; i < df.c; i++) {
            XtX[i * num_vars + k] /= XtX[k * num_vars + k];
        }

        for (int i = k + 1; i < num_vars; i++) {
            for (int j = k + 1; j < num_vars; j++) {
                XtX[i * num_vars + j] -= XtX[i * num_vars+ k] * XtX[k * num_vars + j];
            }
        }
    }

    double* inv = new double[num_vars * num_vars];

    for (int col = 0; col < num_vars; col++) {

        double* y = new double[num_vars];

        for (int i = 0; i <  num_vars; i++) {
            y[i] = (i == col) ? 1.0 : 0.0;

            for (int j = 0; j < i; j++) {
                y[i] -= XtX[i * num_vars + j] * y[j];
            }
        }

        for (int i = num_vars - 1; i >= 0; i--) {
            inv[i * num_vars + col] = y[i];

            for (int j = i + 1; j < num_vars; j++) {
                inv[i * num_vars + col] -= XtX[i * num_vars + j] * inv[j * num_vars + col];
            }

            inv[i * num_vars + col] /= XtX[i * num_vars + i];
        }

        delete[] y;
    }

    // final betas 

    for (int i = 0; i < num_vars; i++) {   // for each beta
        double sum = 0.0;

        for (int j = 0; j < num_vars; j++) {
            sum += inv[i * num_vars + j] * Xty[j];
        }

        df.beta_hat[i] = sum;
    }

    cout << "\nInverse X^tX" << endl;

    for (int i = 0; i < num_vars; i++) {
        for (int j = 0; j < num_vars; j++) {
            cout << inv[i * num_vars + j] << " ";
        }
        cout << endl;
    }

    cout << "\nX^ty" << endl;

    for (int i = 0; i < num_vars; i++) {
        for (int j = 0; j < num_vars; j++) {
            cout << Xty[i * num_vars + j] << " ";
        }
        cout << endl;
    }

    cout << "\nBeta Coefficients: " << endl;

    for (int i = 0; i < num_vars; i++){
        cout << df.beta_hat[i] << endl;
    }

    delete[] XtX;
    delete[] Xty;
    delete[] inv;

    XtX = nullptr;
    Xty = nullptr;
    inv = nullptr;

}


double LinearRegression::beta0(double x_mean, double y_mean, double beta1) {

    double beta0 = y_mean - (beta1 * x_mean);

    return beta0;
}

double LinearRegression::sse(double y[], double y_hat[], int len){
    
    double sse = 0.0;

    for (int i = 0; i < len; i++) {
        sse += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }

    return sse; 
}