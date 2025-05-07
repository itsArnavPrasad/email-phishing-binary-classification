# 📧 Phishing Email Detector using Machine Learning

This project applies multiple machine learning models to classify emails as phishing or non-phishing. It highlights challenges in handling imbalanced datasets and evaluates models based on precision, recall, F1-score, and confusion matrices.

## 📂 Dataset

- **Source**: [Email Phishing Dataset (Kaggle)](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset)
- **Description**: A large-scale dataset containing labeled emails for phishing classification. The dataset is highly imbalanced with a significantly larger number of non-phishing emails (class 0).

## 📌 Models Evaluated

1. **Logistic Regression**
2. **Decision Tree (depths 10, 30, 40, 70)**
3. **Support Vector Machine (SVM & LinearSVC)**
4. **Random Forest (with and without RandomSearchCV tuning)**
5. **K-Nearest Neighbors (KNN, k=1 to 20)**

## 📊 Performance Summary

| Model               | Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score (Phishing) |
|---------------------|----------|-----------------------|--------------------|----------------------|
| Logistic Regression | 0.5349   | 0.0217                | 0.7755             | 0.0423               |
| Decision Tree (D=40)| 0.98     | 0.27                  | 0.36               | 0.31                 |
| SVM (with SMOTE)    | 0.5887   | 0.0214                | 0.6729             | 0.0415               |
| Random Forest       | 0.98     | 0.37                  | 0.37               | 0.37                 |
| Random Forest (Tuned)| 0.9852  | 0.43                  | 0.35               | 0.38                 |
| KNN (k=1)           | 0.9751   | 0.23                  | 0.38               | 0.29                 |

> **Note**: Class 1 = Phishing, Class 0 = Non-phishing

## 🧠 Key Learnings

- **Imbalanced Data**: Precision for phishing emails was consistently low. Techniques like SMOTE, class weighting, and careful model tuning were critical.
- **Model Behavior**:
  - Logistic Regression and SVM had high recall but low precision.
  - Decision Trees and Random Forests offered better balance at higher computational costs.
  - KNN trained fast and showed promising results at k=1.
- **Tuning & Optimization**:
  - Random Forest with `RandomSearchCV` achieved the best F1-score.
  - GridSearchCV was avoided due to computational cost on large datasets.
- **Computational Efficiency**:
  - Decision Trees and KNN trained faster than SVM and Random Forests.

## 🔧 Tools & Techniques

- **Python**
- **scikit-learn**
- **SMOTE** (from `imblearn`)
- **Pandas, NumPy, Matplotlib, Seaborn**


## 📈 Future Work
- Try deep learning models (e.g., LSTM, BERT for text features).

- Build an ensemble using stacking or boosting.

- Explore additional features and email content-based analysis.

