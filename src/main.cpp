#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

// y_hat prediction 
void predict_y(double x[], double y_hat[], int len, double beta0, double beta1){

    for (int i = 0; i < len; i++) {
        y_hat[i] = beta0 + (beta1 * x[i]);
    }

}

// All calcs for stats and reg coeff
double calc_mean(double arr[], int len){
    double arr_sum = 0.0;

    for (int i=0;i<len;i++){
        arr_sum += arr[i];
    }
    double arr_mean = (arr_sum / len);
    return arr_mean;
} 

double calc_variance(double arr[],int len, double mean) {
    
    double var = 0.0;

    for(int i=0;i<len;i++){
        double n_diff = arr[i] - mean;
        var += (n_diff * n_diff);
    }

    return var;
}

double calc_covariance(double x[], double y[], int len, double x_mean, double y_mean) {

    double sum_x_y = 0.0;

    for (int i = 0; i < len; i++) {
        double x_diff = x[i] - x_mean; 
        double y_diff = y[i] - y_mean; 
              
        sum_x_y += x_diff * y_diff;
    }

    return sum_x_y ;
}

double calc_beta1(double cov_xy, double var_x){

    double beta1 = cov_xy / var_x;       

    return beta1;
}

double calc_beta0(double x_mean, double y_mean, double beta1) {

    double beta0 = y_mean - (beta1 * x_mean);

    return beta0;
}

double calc_sse(double y[], double y_hat[], int len){
    
    double sse = 0.0;

    for (int i = 0; i < len; i++) {
        sse += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }

    return sse; 
}

int main() {

    //Read .csv file (static)
    int loc = 0;
    int i = 0;
    int N = 0; 
    
    ifstream iFile;
    string line = "", first = "", last = "";
    iFile.open("tips.csv");
    getline(iFile,line);

    while (getline(iFile, line)){ 
        N++;
    }
    
    cout << "Number of rows in the dataset: " << N << endl; 

    iFile.close();

    iFile.open("tips.csv");
    getline(iFile,line);

    double* x = new double[N]; 
    double* y = new double[N]; 

    while (getline(iFile,line) && i < N){

        loc = line.find(",");
        first = line.substr(0, loc);
        line = line.substr(loc + 1, line.length());

        loc = line.find(",");
        last = line.substr(0, loc);
        line = line.substr(loc + 1, line.length());
        
        x[i] = stod(first);
        y[i] = stod(last);

        i++;
    }

    iFile.close();

    if (N <= 0){
        cout << "Number of rows should be greated than 0" << endl;
        return 1;
    }

    //Inititalizing Accumulators 
    int len_x = N;
    int len_y = N;

    // dynamically allocated arrays 
    double* y_hat = new double[N];
    
    cout << "Welcome to Machine Learning in C++" << endl;
    
    // means
    double x_mean = calc_mean(x, len_x);
    double y_mean = calc_mean(y, len_y);

    //variance
    double x_var = calc_variance(x, len_x, x_mean);
    double y_var = calc_variance(y, len_y, y_mean);

    //sum of difference product (x, y)
    double sum_x_y = calc_covariance(x, y, len_x, x_mean, y_mean);
    
    //regression coefficients 
    double beta_1 = calc_beta1(sum_x_y, x_var);
    double beta_0 = calc_beta0(x_mean, y_mean, beta_1);

    //y_hat 
    predict_y(x, y_hat, len_x, beta_0, beta_1);

    //mse 
    double sse = calc_sse(y, y_hat, len_y);
    
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
    
    // garbage memory cleanup
    delete[] x;
    delete[] y;
    delete[] y_hat;

    x = nullptr;
    y = nullptr; 
    y_hat= nullptr;

    return 0;
}