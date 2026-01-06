#include <iostream>
#include <cmath>
using namespace std;

int main() {

    //Inititalizing Accumulators 

    float x[10];
    float y[10];
    
    int len_x = sizeof(x) / sizeof(x[0]);
    int len_y = sizeof(y) / sizeof(y[0]);
    
    float x_sum = 0.0f;
    float y_sum = 0.0f;
    
    float x_var = 0.0f;
    float y_var = 0.0f;
    
    float sum_x_y = 0.0f;  // co-var calc 
    float er = 0.0f;
    
    float y_hat[10] = {};
    
    cout << "Welcome to Machine Learning in C++" << endl;
    
    cout << "Please enter 10 values of X:" << endl;
    for (int i = 0; i < len_x; i++) {
        cin >> x[i];
    }
    
    cout << "\nPlease enter 10 values of Y:" << endl;

    // Aggregation 
    //sums
    for (int i = 0; i < len_y; i++) {
        cin >> y[i];
    }
    
    for (int i = 0; i < 10; i++) {
        x_sum += x[i];
        y_sum += y[i];
    }
    
    //means
    float x_mean = x_sum / len_x;
    float y_mean = y_sum / len_y;
    

    // var and co-var components
    for (int i = 0; i < 10; i++) {
        float x_diff = x[i] - x_mean; 
        float y_diff = y[i] - y_mean; 
        
        x_var += (x_diff * x_diff);      //sum of square diff of x    
        y_var += (y_diff * y_diff);      //sum of square diff of y   
        sum_x_y += x_diff * y_diff;      //sum of product of differences (x,y)
    }
    
    // regresssion coefficients
    float beta_1 = sum_x_y / x_var;       
    float beta_0 = y_mean - (beta_1 * x_mean);
    
    //predict y 
    for (int i = 0; i < 10; i++) {
        y_hat[i] = beta_0 + (beta_1 * x[i]);
    }
    //error
    for (int i = 0; i < 10; i++) {
        er += pow(y[i] - y_hat[i], 2);
    }
    
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
    
    cout << "\nMean Square Error (MSE): " << er / len_y << endl;
    cout << "Root Mean Square Error (RMSE): " << sqrt(er / len_y) << endl;
    
    return 0;
}