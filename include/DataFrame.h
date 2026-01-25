#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
using namespace std;

class DataFrame {

private:
    string path;

    void get_cols();
    int count_cols();
    int count_rows();

public:
    double* data;
    double* x;
    double* y;
    double* beta_hat;

    int* x_id;
    string* cols;

    int n;      
    int c;      
    int x_len;  
    int y_id;

    DataFrame(const string& path);
    ~DataFrame();

    void read_csv();
    void select_cols(string dcols[], string dcol, int len_x);
    void head(int r = 5);
    void tail(int r = 5);
};

#endif
