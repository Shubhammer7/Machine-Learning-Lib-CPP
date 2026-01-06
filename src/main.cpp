#include <iostream>
#include <cmath>
using namespace std;

//13.9970629 11.55706149 11.50305058 9.26796272 7.58230226 11.24794329 9.36474558 9.44889943 4.91445624 10.43529344

//13.56090341 12.59844447 12.22159385 8.17066895 6.92807306 13.6060403 9.67835263 8.52601592 4.34361701 8.4418719

// Const for testing calcs 
const int N = 10;

// y_hat prediction 
void predict_y(float x[], float y_hat[], int len, float beta0, float beta1){

    for (int i = 0; i < len; i++) {
        y_hat[i] = beta0 + (beta1 * x[i]);
    }

}

// All calcs for stats and reg coeff
float calc_mean(float arr[], int len){
    float arr_sum = 0.0f;

    for (int i=0;i<len;i++){
        arr_sum += arr[i];
    }
    float arr_mean = (arr_sum / len);
    return arr_mean;
} 

float calc_variance(float arr[],int len, float mean) {
    
    float var = 0.0f;

    for(int i=0;i<len;i++){
        float n_diff = arr[i] - mean;
        var += (n_diff * n_diff);
    }

    return var;
}

float calc_covariance(float x[], float y[], int len, float x_mean, float y_mean) {

    float sum_x_y = 0.0f;

    for (int i = 0; i < len; i++) {
        float x_diff = x[i] - x_mean; 
        float y_diff = y[i] - y_mean; 
              
        sum_x_y += x_diff * y_diff;
    }

    return sum_x_y ;
}

float calc_beta1(float cov_xy, float var_x){

    float beta1 = cov_xy / var_x;       

    return beta1;
}

float calc_beta0(float x_mean, float y_mean, float beta1) {

    float beta0 = y_mean - (beta1 * x_mean);

    return beta0;
}

float calc_sse(float y[], float y_hat[], int len){
    
    float sse = 0.0f;

    for (int i = 0; i < len; i++) {
        sse += pow(y[i] - y_hat[i], 2);
    }

    return sse; 
}

int main() {

    //Inititalizing Accumulators 
    // float* x; 

    float y[N];
    
    float* x = new float[N];

    int len_x = N;
    int len_y = N;

    if (len_x == 0){
        cout << "Length of the input array to be greated than 0" << endl;
    }
    
    if (len_y == 0){
        cout << "Length of the input array needs to be greated than 0" << endl;
    }

    float y_hat[N] = {};
    
    cout << "Welcome to Machine Learning in C++" << endl;
    
    // Read input arrays (only X, y. SLR so far)
    cout << "Please enter 10 values of X:" << endl;
    for (int i = 0; i < len_x; i++) {
        cin >> x[i];
    }
    
    cout << "\nPlease enter 10 values of Y:" << endl;

    for (int i = 0; i < len_y; i++) {
        cin >> y[i];
    }
    
    // means
    float x_mean = calc_mean(x, len_x);
    float y_mean = calc_mean(y, len_y);

    //variance
    float x_var = calc_variance(x, len_x, x_mean);
    float y_var = calc_variance(y, len_y, y_mean);

    //sum of difference product (x, y)
    float sum_x_y = calc_covariance(x, y, len_x, x_mean, y_mean);
    
    //regression coefficients 
    float beta_1 = calc_beta1(sum_x_y, x_var);
    float beta_0 = calc_beta0(x_mean, y_mean, beta_1);

    //y_hat 
    predict_y(x, y_hat, len_x, beta_0, beta_1);

    //mse 
    float sse = calc_sse(y, y_hat, len_y);
    
    // output results
    cout << "\n---------------Summary Statistics---------------\n" << endl;
    cout << "Mean of X: " << x_mean << endl;
    cout << "Variance of X: " << x_var / (len_x - 1) << " (sample)" << endl;
    cout << "Standard Deviation of X: " << sqrt(x_var / (len_x - 1)) << " (sample)" << endl;
    
    cout << "\nMean of Y: " << y_mean << endl;
    cout << "Variance of Y: " << y_var / (len_y - 1) << " (sample)" << endl;
    cout << "Standard Deviation of Y: " << sqrt(y_var / (len_y - 1)) << " (sample)" << endl;
    
    cout << "\nCovariance of (X,Y): " << sum_x_y / (len_x - 1) << " (sample)" << endl;
    
    cout << "\nRegression Coefficient β₁: " << beta_1 << endl;
    cout << "Intercept β₀: " << beta_0 << endl;
    
    cout << "\nMean Square Error (MSE): " << sse / len_y << endl;
    cout << "Root Mean Square Error (RMSE): " << sqrt(sse / len_y) << endl;
    

    delete[] x;
    x = nullptr;

    return 0;
}