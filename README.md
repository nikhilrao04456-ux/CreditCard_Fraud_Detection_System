# CreditCard_Fraud_Detection_System
Machine learning model to detect fraudulent credit card transactions using Random Forest on a highly imbalanced real-world dataset.


# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions
using classification algorithms on a real-world imbalanced dataset.

---

## About the Project

Credit card fraud is a major problem where fraudulent transactions are
extremely rare compared to normal ones. This project builds a machine
learning model that can correctly identify fraud even when the dataset
is heavily imbalanced (only 0.17% of transactions are fraud).

Dataset: Kaggle Credit Card Fraud Detection
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Total Transactions: 284,807
Fraud Cases: 492 (0.17%)

---

## What This Project Does

1. Loads and explores the dataset (shape, missing values, class distribution)
2. Visualizes class imbalance, amount distribution, and correlation heatmap
3. Preprocesses data — applies log transformation on Amount column
4. Trains a baseline Decision Tree model
5. Compares three models using Stratified K-Fold cross-validation
6. Selects Random Forest as the best model
7. Tunes hyperparameters using RandomizedSearchCV
8. Evaluates final model using PR-AUC, confusion matrix, classification report
9. Builds a prediction function to classify new transactions

---

## Models Used

- Decision Tree Classifier
- Histogram Gradient Boosting Classifier
- Random Forest Classifier (Best Model)

---

## Why PR-AUC and not Accuracy?

Because the dataset is highly imbalanced, accuracy is misleading.
A model that predicts everything as non-fraud would still get 99.8% accuracy.
PR-AUC (Precision-Recall Area Under Curve) is a better metric for
imbalanced classification problems like fraud detection.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## How to Run

1. Clone the repository
   git clone https://github.com/nikhilrao04456-ux/credit-card-fraud-detection

2. Install required libraries
   pip install pandas numpy matplotlib seaborn scikit-learn

3. Download the dataset from Kaggle and place creditcard.csv
   in the project folder

4. Open the notebook
   jupyter notebook 15_3_credit_card_fraud_detection.ipynb

5. Run all cells

---

## Project Structure

credit-card-fraud-detection/
│
├── 15_3_credit_card_fraud_detection.ipynb   # Main notebook
├── creditcard.csv                            # Dataset (download from Kaggle)
└── README.md                                 # Project documentation

---

## Results

- Best Model: Random Forest Classifier
- Evaluation Metric: PR-AUC (Precision-Recall AUC)
- Class Imbalance handled using: class_weight = balanced_subsample
- Hyperparameter tuning: RandomizedSearchCV with Stratified K-Fold

---

## Author

Nikhil Yadav
B.E. Electronics and Electrical Engineering — MBM University, Jodhpur
GitHub: https://github.com/nikhilrao04456-ux
LinkedIn: https://www.linkedin.com/in/nikhil-yadav-37281132a
