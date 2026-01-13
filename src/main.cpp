#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

struct Dataset {
    double* x;
    double* y;
    int n;

    Dataset(int size) {
        n = size;
        x = new double[n];
        y = new double[n];
    }

    ~Dataset() {
        delete[] x;
        delete[] y;
    }
};

struct Predictions {
    double* y_hat;
    int n;

    Predictions(int size){
        n = size;
        y_hat = new double[n];        
    }

    ~Predictions() {
        delete[] y_hat;
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


void read_csv(const string& path, Dataset& data) {
    ifstream file(path);
    string line;
    getline(file, line); 

    int i = 0;
    while (getline(file, line) && i < data.n) {
        int loc = line.find(",");
        double x_val = stod(line.substr(0, loc));
        double y_val = stod(line.substr(loc + 1));

        data.x[i] = x_val;
        data.y[i] = y_val;
        i++;
    }
}

// y_hat prediction 
void predict_y(double x[], double y_hat[], int len, double beta0, double beta1){

    for (int i = 0; i < len; i++) {
        y_hat[i] = beta0 + (beta1 * x[i]);
    }

}
RegressionResults train(const Dataset& data, Predictions& preds) {

    RegressionResults res;

    int len_x = data.n;
    int len_y = data.n;
    
    // means
    res.x_mean = calc_mean(data.x, len_x);
    res.y_mean = calc_mean(data.y, len_y);

    //variance
    res.ssx = calc_ss(data.x, len_x, res.x_mean);
    res.ssy = calc_ss(data.y, len_y, res.y_mean);

    //sum of difference product (x, y)
    res.sum_x_y = calc_covariance(data.x, data.y, len_x, res.x_mean, res.y_mean);
    
    //regression coefficients 
    res.beta_1 = calc_beta1(res.sum_x_y, res.ssx);
    res.beta_0 = calc_beta0(res.x_mean, res.y_mean, res.beta_1);

    predict_y(data.x, preds.y_hat, len_x, res.beta_0, res.beta_1);

    //mse 
    res.sse = calc_sse(data.y, preds.y_hat, len_y);

    //mae
    res.mae = calc_mae(data.y, preds.y_hat, len_y);

    //r_squared
    res.r_squared = calc_r_squared(res.ssy, res.sse);

    return res;

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

double calc_ss(double arr[],int len, double mean) {
    
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

double calc_mae(double y[], double y_pred[], int len){

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

double calc_r_squared(double sst, double ssr){

    double r_squared = 0.0;

    r_squared = 1 - (ssr / sst);
    
    return r_squared; 

}






int count_rows(string path) {

    int n = 0; 
    int loc = 0;
    int i = 0;

    ifstream iFile;
    string line = "", first = "", last = "";
    iFile.open(path);
    getline(iFile,line);

    while (getline(iFile, line)){ 
        n++;
    }
    
    iFile.close();

    return n;
}


int main() {   

    int n = count_rows("tips.csv");
    if (n <= 0){
        cout << "Number of rows should be greated than 0" << endl;
        return 1;
    }

    Dataset data(n); 
    Predictions preds(n);
    read_csv("tips.csv", data);

    int loc = 0;
    int i = 0;
    int len_x = data.n;
    int len_y = data.n;

    
    cout << "Welcome to Machine Learning in C++" << endl;
    
    RegressionResults res = train(data,preds);
    
    // output results
    cout << "\n---------------Summary Statistics---------------\n" << endl;
    cout << "Mean of X: " << res.x_mean << endl;
    cout << "Variance of X: " << res.ssx / (len_x - 1) << " (sample)" << endl;
    cout << "Standard Deviation of X: " << sqrt(res.ssx / (len_x - 1)) << " (sample)" << endl;
    
    cout << "\nMean of Y: " << res.y_mean << endl;
    cout << "Variance of Y: " << res.ssy / (len_y - 1) << " (sample)" << endl;
    cout << "Standard Deviation of Y: " << sqrt(res.beta_1 / (len_y - 1)) << " (sample)" << endl;
    
    cout << "\nCovariance of (X,Y): " << res.sum_x_y / (len_x - 1) << " (sample)" << endl;
    
    cout << "\nRegression Coefficient β₁: " << res.beta_1 << endl;
    cout << "Intercept β₀: " << res.beta_0 << endl;
    
    cout << "\nMean Square Error (MSE): " << res.sse / len_y << endl;
    cout << "Root Mean Square Error (RMSE): " << sqrt(res.sse / len_y) << endl;
    cout << "Mean Absolute Error (MAE): " << res.mae << endl;
    cout << "Coeffecient of Determination (R^2): "<< res.r_squared << endl;

    return 0;
}