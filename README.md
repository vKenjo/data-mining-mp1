# Data Mining MP1: Weather Data Analysis

This repository contains the implementation for Machine Problem 1 for a Data Mining course. The project performs various data mining calculations on a provided weather dataset (`data1.csv`), which has been decoded into numeric values (`decoded.csv`).

## 📊 Dataset Overview

The dataset contains 70 data points with 5 features:

* **Outlook**: `sunny` (1), `overcast` (2), `rainy` (3)
* **Temperature**: `hot` (1), `mild` (2), `cool` (3)
* **Humidity**: `high` (1), `normal` (2)
* **Windy**: `TRUE` (1), `FALSE` (2)
* **Play** (Target): `yes` (1), `no` (2)

## 🎯 Solved Problems

The exact step-by-step solutions for the following 5 problems are implemented in Python:

1. **Distance Metrics**: Calculates Euclidean Distance, Manhattan Distance, Minkowski Distance (r=5), and Chebyshev/Max Distance (r=∞) between the dataset mean and specific data points (7 and 70).
2. **Mahalanobis Distance**: Computes the Mahalanobis distance from a test point `(overcast, hot, normal)` using the first 8 rows of the dataset. Includes detailed steps for mean vector, centered data, covariance matrix, and inverse covariance matrix.
3. **Similarity Measures**: Calculates both the Cosine Similarity and Extended Jaccard Similarity between data point 20 and data point 27.
4. **Entropy**: Calculates the Shannon entropy for both Temperature and Humidity attributes across the entire dataset, including maximum possible entropy for their respective classes.
5. **Mutual Information**: Computes the mutual information `I(Outlook; Temperature)` using marginal distributions, joint distributions, and joint entropy. Confirmed calculation using two different standardized mathematical definitions.

## 🚀 How to Run

There are two main ways to run this project:

### 1. Python Script (Terminal Output)

You can run the full script which will print out a detailed, step-by-step mathematical breakdown for all 5 problems to your console.

```bash
PYTHONPATH=src python -c "import mp1; mp1.main()"
# OR simply:
python src/mp1/main.py
```

### 2. Jupyter Notebook

An interactive Jupyter Notebook (`mp1.ipynb`) is provided for easier testing and exploration.

## 📁 Repository Structure

```text
data-mining-mp1/
├── README.md                # Project documentation
├── mp1.ipynb                # Interactive Jupyter Notebook version
├── data/
│   ├── data1.csv            # Original raw dataset
│   └── decoded.csv          # Decoded numeric dataset used for calculations
└── src/
    └── mp1/
        ├── __init__.py      # Package initialization
        └── main.py          # Main computation script solving the 5 problems
```

## 🛠️ Tools Used

* **Language**: Python 3.13+
* **Core Libraries**: `csv`, `math` (No external data science libraries like pandas or numpy were used to show the hand-calculated nature of the data mining algorithms).
