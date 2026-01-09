# Machine Learning Library in C++ (From Scratch)

This project is a from-scratch implementation of **Simple Linear Regression (SLR)** in C++, built to deeply understand the mathematics, memory management, and program structure behind machine learning libraries.

The goal of this repository is **learning**, not performance or feature completeness.

---

## What is implemented so far

- Reading numerical data from a CSV file using standard C++ (`ifstream`)
- Dynamic memory allocation for datasets
- Simple Linear Regression using the closed-form solution
- Calculation of:
  - Mean
  - Variance
  - Covariance
  - Regression coefficients (β₀, β₁)
- Prediction (`y_hat`)
- Evaluation metrics:
  - MSE
  - RMSE
  - MAE
  - R²
- Explicit memory cleanup (`new[]` / `delete[]`)

No external libraries or STL containers are used for the core logic.

---

## Project philosophy

This project intentionally avoids:
- High-level ML libraries
- Abstractions that hide memory ownership
- Premature object-oriented design

The focus is on:
- Understanding how regression works numerically
- Learning how data is stored and manipulated in memory
- Writing predictable, explicit C+
