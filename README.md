# 💳 Credit Card Fraud Detection using K-Nearest Neighbors (KNN)

This project applies the **K-Nearest Neighbors (KNN)** algorithm to detect fraudulent transactions in a real-world, highly imbalanced credit card dataset. It includes preprocessing, scaling, tuning the value of `k`, evaluating classification performance, and visualizing results using PCA.

---
![Project Thumbnail](credit_card_fraud-thumbnail.png)

## 📌 Project Objectives

- Detect fraudulent transactions using KNN.
- Handle an imbalanced dataset with only ~0.17% fraud cases.
- Tune the number of neighbors (`k`) to optimize model performance.
- Visualize model predictions and compare with true labels.

---

## 📂 Dataset Overview

- **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Frauds:** 492 (≈0.17%)
- **Features:**  
  - 28 PCA-transformed components (`V1` to `V28`)
  - `Time` (seconds since first transaction)
  - `Amount` (transaction value)
  - `Class` (0 = non-fraud, 1 = fraud)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🔍 Key Steps Performed

### 1. **Data Preprocessing**
- Dropped any missing values in the `Class` column.
- Scaled `Time` and `Amount` using `StandardScaler`.

### 2. **Train-Test Split**
- Used `train_test_split` with `test_size = 0.2` and `stratify=y`.

### 3. **Modeling with KNN**
- Trained KNN classifier with `k = 5`.
- Tuned `k` from 1 to 20 to identify the best-performing value.

### 4. **Model Evaluation**
- Used `accuracy_score` and `classification_report`.
- Confusion matrix to visualize TP, FP, TN, FN.
- Best performance at `k = 3`.

### 5. **Visualization**
- Reduced feature space to 2D using PCA.
- Compared:
  - Actual labels
  - Predictions with `k = 3`
  - Predictions with `k = 5`

---

## 📈 Results

### ✅ Best Model: `K = 3`

| Metric          | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|-----------------|---------------------|------------------|
| Precision       | 1.00                | 0.92             |
| Recall          | 1.00                | 0.83             |
| F1-score        | 1.00                | 0.87             |
| Accuracy        | **99.96%**          |                  |

### 🧾 Confusion Matrix

|                  | Predicted: Non-Fraud | Predicted: Fraud |
|------------------|----------------------|------------------|
| Actual: Non-Fraud| 56857 (TN)           | 7 (FP)           |
| Actual: Fraud    | 19 (FN)              | 79 (TP)          |

---

## 🔬 Observations & Insights

- High overall accuracy is misleading due to class imbalance.
- KNN achieved **83% recall on frauds** — a solid baseline.
- False negatives (missed frauds) are still a concern.
- Scaling and proper tuning significantly impacted results.
- KNN is slow to evaluate on large datasets due to its lazy-learning nature.

---

## 📌 Conclusion

KNN is a simple yet effective **baseline model** for fraud detection. It works well with proper feature scaling and hyperparameter tuning but struggles with severe class imbalance. For production systems, more advanced models like **Random Forest**, **XGBoost**, or **SMOTE-based approaches** are recommended for better fraud recall.

---

## 📁 Files in this Repo

| File                      | Description                            |
|---------------------------|----------------------------------------|
| `fraud_detection_knn.ipynb` | Main Jupyter Notebook                  |
| `fraud_detection_knn.html` | Rendered version of the notebook       |
| `README.md`               | This file                              |

---

## 🤝 Connect

- [LinkedIn](https://www.linkedin.com/in/varsha-shekhar)
- 📧 varshaiyer96@gmail.com
