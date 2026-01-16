# Machine Learning Library in C++ (From Scratch)

This project is a **from-scratch implementation of Simple Linear Regression (SLR)** in C++, built to deeply understand the **mathematics, numerical stability, memory management, and program structure** behind machine learning libraries.

The goal of this repository is **learning and correctness**, not performance benchmarks or feature completeness.

---

## What is implemented so far

- Reading numerical data from CSV files using standard C++ (`ifstream`)
- Dynamic dataset sizing via runtime row counting
- Manual memory management using heap allocation (`new[]` / `delete[]`)
- Simple Linear Regression using the closed-form solution
- Explicit computation of:
  - Mean
  - (Unnormalized) variance and covariance
  - Regression coefficients (β₀, β₁)
- Prediction (`y_hat`)
- Evaluation metrics:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Coefficient of Determination (R²)
- Separation of concerns via pure functions
- RAII-style ownership using destructors for dataset memory
- Numerical stability awareness (comparison of naive vs compensated summation)

No external libraries or STL containers are used for the **core numerical logic**.

---

## Project philosophy

This project intentionally avoids:

- High-level ML libraries (e.g. sklearn, Eigen, BLAS)
- Abstractions that hide memory ownership
- Premature object-oriented design
- “Black-box” numerical routines

The focus is on:

- Understanding how regression works **numerically**, not just algebraically
- Learning how data is stored, accessed, and freed in memory
- Writing predictable, explicit C++ with clear ownership semantics
- Developing intuition for numerical issues such as floating-point precision and summation error
- Building ML functionality bottom-up, the way core libraries are designed

---

## What this project is **not**

- A production-ready ML framework
- An optimized or vectorized implementation
- A replacement for existing ML libraries

This repository prioritizes **clarity over cleverness** and **correctness over convenience**.

---

## Current scope

- Single-feature (univariate) linear regression
- Batch training on in-memory datasets
- CPU-only, sequential execution

---

## Planned extensions

- Numerically stable summation (Kahan / compensated summation) across all aggregations
- Clear separation between training and inference
- Multiple Linear Regression (design → implementation)
- Explicit matrix memory layouts (row-major)
- Improved input validation and error handling
- Clean public API resembling minimal sklearn-style usage
- Optional comparison against standard ML libraries for verification

---

## Why C++?

C++ forces explicit thinking about:

- Memory allocation and lifetime
- Data layout
- Performance vs correctness tradeoffs
- Numerical stability

By removing abstraction layers, this project aims to build **first-principles understanding** of how machine learning systems work under the hood.
