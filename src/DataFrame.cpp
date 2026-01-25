#include "../include/DataFrame.h"
#include <iostream>
#include <fstream>
using namespace std;

DataFrame::DataFrame(const string& file_path) {

    path = file_path;

    n = count_rows();
    c = count_cols();

    data = new double[n * c];
    cols = new string[c];

    x = nullptr;
    y = nullptr;
    beta_hat = nullptr;
    x_id = nullptr;

    get_cols();
    read_csv();
}

DataFrame::~DataFrame() {
    delete[] data;
    delete[] cols;
    delete[] x;
    delete[] y;
    delete[] beta_hat;
    delete[] x_id;
}


void DataFrame::read_csv() {
    ifstream file(path);
    string line;

    getline(file, line); 

    int row = 0;
    while (getline(file, line) && row < n) {

        int col = 0;
        size_t start = 0;

        while (col < c) {

            size_t comma = line.find(',', start);
            string value;

            if (comma == string::npos) {
                value = line.substr(start);
            } else {
                value = line.substr(start, comma - start);
            }

            while (!value.empty() && (value.back() == '\r' || value.back() == ' '))
                value.pop_back();

            if (value == "")
                data[row * c + col] = 0.0;
            else
                data[row * c + col] = stod(value);

            if (comma == string::npos) break;

            start = comma + 1;
            col++;
        }

        row++;
    }

    file.close();
}


void DataFrame::get_cols() {
    ifstream file(path);
    string line;

    getline(file, line);
    file.close();

    int i = 0;
    size_t pos;

    while ((pos = line.find(',')) != string::npos) {
        cols[i++] = line.substr(0, pos);
        line.erase(0, pos + 1);
    }
    cols[i] = line;
}

void DataFrame::select_cols(string dcols[], string dcol, int len_x) {

    x_len = len_x;

    x_id = new int[x_len];
    x = new double[n * x_len];
    y = new double[n];
    beta_hat = new double[x_len];

    for (int i = 0; i < x_len; i++) {
        for (int j = 0; j < c; j++) {
            if (dcols[i] == cols[j])
                x_id[i] = j;
            if (dcol == cols[j])
                y_id = j;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < x_len; j++)
            x[i * x_len + j] = data[i * c + x_id[j]];

        y[i] = data[i * c + y_id];
    }
}

void DataFrame::head(int r) {
    for (int i = 0; i < r && i < n; i++) {
        for (int j = 0; j < c; j++)
            cout << data[i * c + j] << " ";
        cout << endl;
    }
}

void DataFrame::tail(int r) {
    for (int i = n - r; i < n; i++) {
        for (int j = 0; j < c; j++)
            cout << data[i * c + j] << " ";
        cout << endl;
    }
}

int DataFrame::count_rows() {
    ifstream file(path);
    string line;
    int rows = 0;

    getline(file, line); 
    while (getline(file, line))
        rows++;

    file.close();
    return rows;
}

int DataFrame::count_cols() {
    ifstream file(path);
    string line;
    int cols = 1;

    getline(file, line);
    for (int i = 0; i < line.size(); i++) {
        if (line[i] == ',') cols++;
    }

    file.close();
    return cols;
}


