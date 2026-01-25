#include "../include/DataFrame.h"

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

int main() {

    DataFrame df("data/diamonds.csv");

    df.head();
    df.tail();


    // RegressionResults res = train(data,preds);
    
    // cout << "\nWelcome to Machine Learning in C++" << endl;

    // cout << "\nColumns in the dataset: " << endl;
    // get_cols(data, path);

    // cout << "\nFirst 5 rows of the dataset: " << endl;
    // head(data);

    // cout << "\nLast 5 rows of the dataset: " << endl;
    // tail(data);

    // cout << "---------------------------------------------------------" << endl;

    // cout << "\nSelected columns (X): " << endl;
    // select_cols(data, X_col, y_col, 3);

    // cout << "---------------------------------------------------------" << endl;

    // cout << "\nInversed data" << endl;
    // calc_beta1(data);
   
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