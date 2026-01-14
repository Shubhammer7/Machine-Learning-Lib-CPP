#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

struct Dataset {
    double* x;
    double* y;
    string* cols;
    int n;
    int c;

    Dataset(int size_rows) {
        n = size_rows;
        // c = size_cols;
        x = new double[n];
        y = new double[n];
    }

    ~Dataset() {
        delete[] x;
        delete[] y;
        delete[] cols;

        x = nullptr;
        y = nullptr;
        cols = nullptr;
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

//Unsafe input — known technical debt
void read_csv(const string& path, Dataset& data) {
    ifstream file(path);

    string line;
    getline(file, line); 
    int loc = 0;
    int i = 0;

    while (i < data.c - 1) { 
        loc = line.find(",");
        data.cols[i] = line.substr(0, loc);
        line = line.substr(loc + 1);  
        i++;
    }

    data.cols[i] = line; 

    if (file) {
        cout << ".csv file is loaded successfully!" <<endl;
        cout << "Number of Rows: " << data.n << endl;
        cout << "Number of Columns: " << data.c << endl;
        

        while (getline(file, line) && i < data.n) {
            int loc = line.find(",");
            double x_val = stod(line.substr(0, loc));
            double y_val = stod(line.substr(loc + 1));

            data.x[i] = x_val;
            data.y[i] = y_val;
            i++;
    }
    } else {
        cout << "failed to read .csv, please check file path!" <<endl;
        return;
    } 
}

void head(const Dataset& data){

    if (data.n < 5) {
        cout << "Dataset has less than 5 rows! " << endl;
        return;
    }

    for (int i = 0; i < 5; i++){
        cout << data.x[i] << " " << data.y[i] << endl;
    }
}

void tail(const Dataset& data){

    if (data.n < 5) {
        cout << "Dataset has less than 5 rows! " << endl;
        return;
    }

    for (int i = (data.n - 5); i < data.n; i++){
        cout << data.x[i] << " " << data.y[i] << endl;
    }
}



// y_hat prediction 
void predict_y(double x[], double y_hat[], int len, double beta0, double beta1){

    for (int i = 0; i < len; i++) {
        y_hat[i] = beta0 + (beta1 * x[i]);
    }

}

void preview_predictions(const Predictions& preds, int n = 5){

    if (n > preds.n) {
        cout << "n cannot be greater than the length of the dataset" << endl;
        return;
    }

    for (int i = 0; i < n; i++ ){
        cout << preds.y_hat[i] << endl;
    }
}

double kahan_sum(double arr[], int len) {

    double sum = 0.0;
    double c = 0.0;         

    for (int i = 0; i < len; i++) {
        double y = arr[i] - c;
        double t = sum + y;
        c = (t - sum) -y; 
        sum = t;
    }

    return sum;
}

// All calcs for stats and reg coeff

double calc_mean(const Dataset& data, double arr[]) {

    double arr_sum = kahan_sum(arr, data.n);

    return arr_sum / data.n;

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

RegressionResults train(const Dataset& data, Predictions& preds) {

    RegressionResults res;

    int len_x = data.n;
    int len_y = data.n;
    
    // means
    res.x_mean = calc_mean(data, data.x);
    res.y_mean = calc_mean(data, data.y);

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

int count_rows(string path) {

    int n = 0; 
    ifstream iFile;

    if (iFile) {
        cout << ".csv file loaded successfully, counting rows!" << endl;
        string line = "", first = "", last = "";
        iFile.open(path);
        getline(iFile,line);

        while (getline(iFile, line)){ 
            n++;
        }
        
        iFile.close();
    }
    else {
        cout << "Failed to count rows, please check file path!" << endl;
    }

    return n;
    
}

int count_cols(string path) {
    int c = 1;
    ifstream file("tips.csv");
    string line;

    getline(file, line);

    for (int i = 0; i < line.size(); i++){
        if (line[i] == ','){
            c++;
        }
    }

    file.close();

    return c;
}

int main() {   
    
    int n = count_rows("tips.csv");
    int c = count_cols("tips.csv");

    Dataset data(n); 
    Predictions preds(n);

    read_csv("tips.csv", data);
   

    int len_x = data.n;
    int len_y = data.n;

    RegressionResults res = train(data,preds);
    
    cout << "Welcome to Machine Learning in C++" << endl;

    cout << "\nFirst 5 rows of the dataset: " << endl;
    head(data);

    cout << "\nLast 5 rows of the dataset: " << endl;
    tail(data);

    // output results
    cout << "\n---------------Summary Statistics---------------\n" << endl;
    cout << "Mean of X: " << res.x_mean << endl;
    cout << "Variance of X: " << res.ssx / (len_x - 1) << " (sample)" << endl;
    cout << "Standard Deviation of X: " << sqrt(res.ssx / (len_x - 1)) << " (sample)" << endl;
    
    cout << "\nMean of Y: " << res.y_mean << endl;
    cout << "Variance of Y: " << res.ssy / (len_y - 1) << " (sample)" << endl;
    cout << "Standard Deviation of Y: " << sqrt(res.ssy / (len_y - 1)) << " (sample)" << endl;
    
    cout << "\nCovariance of (X,Y): " << res.sum_x_y / (len_x - 1) << " (sample)" << endl;
    
    cout << "\nRegression Coefficient β₁: " << res.beta_1 << endl;
    cout << "Intercept β₀: " << res.beta_0 << endl;

    cout << "\nPredictions: " << endl;
    preview_predictions(preds, 5);

    cout << "\nMean Square Error (MSE): " << res.sse / len_y << endl;
    cout << "Root Mean Square Error (RMSE): " << sqrt(res.sse / len_y) << endl;
    cout << "Mean Absolute Error (MAE): " << res.mae << endl;
    cout << "Coeffecient of Determination (R^2): "<< res.r_squared << endl;

    return 0;
}