#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

struct Dataset {
    double* data;
    double* y;
    int n;
    int c;
    int* x_id;
    int x_len;
    int y_id;
    string* cols;
    double* beta_hat;

    Dataset(int size_rows, int size_cols) {
        n = size_rows;
        c = size_cols;
        data = new double[n * c];
        y = new double[n];
        x_id = new int[c - 1];
        beta_hat = new double[c];
        cols = new string[c];
    }

    ~Dataset() {
        delete[] data;
        delete[] cols;
        delete[] beta_hat;
        delete[] x_id;

        data = nullptr;
        y = nullptr;
        cols = nullptr;
        beta_hat = nullptr;
        x_id = nullptr;
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

void read_csv(const string& path, Dataset& df) {

    ifstream file(path);
    string line;
    getline(file, line); 

    int row = 0;

    while (getline(file, line) && row < df.n) {
        int col = 0;
        size_t start = 0;

        while (col < df.c) {
            size_t comma = line.find(',', start);
            string value = line.substr(start, comma - start);

            df.data[row * df.c + col] = stod(value);

            if (comma == string::npos) break;
            start = comma + 1;
            col++;
        }

        row++;
    }

}

void get_cols(Dataset& data, const string& path) {

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

    for (int i = 0; i < data.c; i++) {
        cout << data.cols[i] << endl;
    }

}
void head(const Dataset& df){

    if (df.n < 5) {
        cout << "Dataset has less than 5 rows! " << endl;
        return;
    }

   for (int i = 0; i < 5; i++) {
        for (int j = 0; j < df.c; j++) {
            cout << df.data[i * df.c + j] << " ";
        }
        cout << endl;
    }
}

void tail(const Dataset& df){

    if (df.n < 5) {
        cout << "Dataset has less than 5 rows! " << endl;
        return;
    }

    for (int i = (df.n - 5); i < df.n; i++){
        for (int j = 0; j < df.c; j++) {
            cout << df.data[i * df.c + j] << " ";
        }
        cout << endl;
    }
}

void select_cols(Dataset& data, string dcols[], string dcol, int len_x){

    if (len_x >= data.c) {
        cout << "Cannot include the entire dataset as the independent variables!" << endl;
    }

    for (int i = 0; i < len_x; i++){
        for (int j = 0; j < data.c; j++){
            if (dcols[i] == data.cols[j]){
                data.x_id[i] = j;
                cout << data.x_id[i] << endl;
            } else if (dcol == data.cols[j]){
                data.y_id = j;
            }
        }
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
        c = (t - sum) - y; 
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

void calc_beta1(const Dataset& df) {

    // calculating the X^TX the hard way, will replace with cholesky soon
    double XtX[df.c * df.c];

    for (int i = 0; i < df.c * df.c; i++) {
        XtX[i] = 0.0;
    } 

    for (int i = 0; i < df.c; i++) {          // column i
        for (int j = 0; j < df.c; j++) {      // column j
            double sum = 0.0;

            for (int k = 0; k < df.n; k++) {  // rows
                sum += df.data[k * df.c + i] * df.data[k * df.c + j];
            }

            XtX[i * df.c + j] = sum;
        }
    }

    // LU decomposition for matrix inverse
    for (int k = 0; k < df.c; k++) {

        for (int i = k + 1; i < df.c; i++) {
            XtX[i * df.c + k] /= XtX[k * df.c + k];
        }

        for (int i = k + 1; i < df.c; i++) {
            for (int j = k + 1; j < df.c; j++) {
                XtX[i * df.c + j] -= XtX[i * df.c + k] * XtX[k * df.c + j];
            }
        }
    }

    double inv[df.c * df.c];

    for (int col = 0; col < df.c; col++) {

        double y[df.c];

        for (int i = 0; i < df.c; i++) {
            y[i] = (i == col) ? 1.0 : 0.0;

            for (int j = 0; j < i; j++) {
                y[i] -= XtX[i * df.c + j] * y[j];
            }
        }

        for (int i = df.c - 1; i >= 0; i--) {
            inv[i * df.c + col] = y[i];

            for (int j = i + 1; j < df.c; j++) {
                inv[i * df.c + col] -= XtX[i * df.c + j] * inv[j * df.c + col];
            }

            inv[i * df.c + col] /= XtX[i * df.c + i];
        }
    }

    cout << "\n(X^T X)^-1\n";
    for (int i = 0; i < df.c; i++) {
        for (int j = 0; j < df.c; j++) {
            cout << inv[i * df.c + j] << " ";
        }
        cout << endl;
    }
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

RegressionResults train(const Dataset& df, Predictions& preds) {

    RegressionResults res;

    int len_x = df.n;
    int len_y = df.n;
    
    // means
    res.x_mean = calc_mean(df, df.data);
    res.y_mean = calc_mean(df, df.y);

    //variance
    res.ssx = calc_ss(df.data, len_x, res.x_mean);
    res.ssy = calc_ss(df.y, len_y, res.y_mean);

    //sum of difference product (x, y)
    res.sum_x_y = calc_covariance(df.data, df.y, len_x, res.x_mean, res.y_mean);
    
    //regression coefficients 
    // res.beta_1 = calc_beta1(res.sum_x_y, res.ssx);
    // res.beta_0 = calc_beta0(res.x_mean, res.y_mean, res.beta_1);

    predict_y(df.data, preds.y_hat, len_x, res.beta_0, res.beta_1);

    //mse 
    res.sse = calc_sse(df.y, preds.y_hat, len_y);

    //mae
    res.mae = calc_mae(df.y, preds.y_hat, len_y);

    //r_squared
    res.r_squared = calc_r_squared(res.ssy, res.sse);

    return res;

}

int count_rows(string path) {

    int n = 0; 
    ifstream iFile;

    if (iFile) {
        cout << "\n.csv file loaded successfully, counting rows!" << endl;
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
    ifstream file(path);
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
    
    string path = "diamonds.csv";
    int n = count_rows(path);
    int c = count_cols(path);

    Dataset data(n,c); 
    Predictions preds(n);

    read_csv(path, data);
   

    int len_x = data.n;
    int len_y = data.n;

    RegressionResults res = train(data,preds);
    
    cout << "\nWelcome to Machine Learning in C++" << endl;

    cout << "\nColumns in the dataset: " << endl;
    get_cols(data, path);

    cout << "\nFirst 5 rows of the dataset: " << endl;
    head(data);

    cout << "\nTransposed Data (head)" << endl;
    calc_beta1(data);

    cout << "\nLast 5 rows of the dataset: " << endl;
    tail(data);



    // output results
    // cout << "\n---------------Summary Statistics---------------\n" << endl;
    // cout << "Mean of X: " << res.x_mean << endl;
    // cout << "Variance of X: " << res.ssx / (len_x - 1) << " (sample)" << endl;
    // cout << "Standard Deviation of X: " << sqrt(res.ssx / (len_x - 1)) << " (sample)" << endl;
    
    // cout << "\nMean of Y: " << res.y_mean << endl;
    // cout << "Variance of Y: " << res.ssy / (len_y - 1) << " (sample)" << endl;
    // cout << "Standard Deviation of Y: " << sqrt(res.ssy / (len_y - 1)) << " (sample)" << endl;
    
    // cout << "\nCovariance of (X,Y): " << res.sum_x_y / (len_x - 1) << " (sample)" << endl;
    
    // cout << "\nRegression Coefficient β₁: " << res.beta_1 << endl;
    // cout << "Intercept β₀: " << res.beta_0 << endl;

    // cout << "\nPredictions: " << endl;
    // preview_predictions(preds, 5);

    // cout << "\nMean Square Error (MSE): " << res.sse / len_y << endl;
    // cout << "Root Mean Square Error (RMSE): " << sqrt(res.sse / len_y) << endl;
    // cout << "Mean Absolute Error (MAE): " << res.mae << endl;
    // cout << "Coeffecient of Determination (R^2): "<< res.r_squared << endl;

    return 0;
}