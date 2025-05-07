# Phishing Email Classification: Model Evaluation Summary

This README summarizes the performance and key learnings from evaluating multiple machine learning models for phishing email classification. The dataset is imbalanced, with significantly more non-phishing (class 0) than phishing (class 1) emails, which posed challenges across all models. Below are the results and insights for each model tested.

Database Used: email-phishing-dataset (https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset?resource=download)
## 1. Logistic Regression
### Learnings
- Required a balanced approach (e.g., class weighting or resampling) to prevent the model from predicting all emails as non-phishing.
- Suffered from a high number of false positives, indicating poor precision for phishing detection.

### Performance Metrics
- **Accuracy**: 0.5349
- **Precision (class 1)**: 0.0217
- **Recall (class 1)**: 0.7755
- **F1-Score (class 1)**: 0.0423
- **Confusion Matrix**:
  ```
  [[82602 72767]
   [  468  1617]]
  ```

## 2. Decision Tree
### Learnings
- Trained quickly, allowing rapid experimentation.
- Performance improved with increased tree depth, but gains plateaued beyond depth 40.
- Higher depths improved precision and accuracy but maintained moderate recall for phishing emails.

### Performance Metrics
- **Depth 10**:
  - Accuracy: 0.62
  - Precision (class 1): 0.02
  - Recall (class 1): 0.70
  - F1-Score (class 1): 0.05
  - Confusion Matrix:
    ```
    [[95977 59392]
     [  631  1454]]
    ```
- **Depth 30**:
  - Accuracy: 0.97
  - Precision (class 1): 0.18
  - Recall (class 1): 0.36
  - F1-Score (class 1): 0.24
  - Confusion Matrix:
    ```
    [[151816  3553]
     [  1328   757]]
    ```
- **Depth 40**:
  - Accuracy: 0.98
  - Precision (class 1): 0.27
  - Recall (class 1): 0.36
  - F1-Score (class 1): 0.31
  - Confusion Matrix:
    ```
    [[153306  2063]
     [  1329   756]]
    ```
- **Depth 70**:
  - Accuracy: 0.98
  - Precision (class 1): 0.27
  - Recall (class 1): 0.36
  - F1-Score (class 1): 0.31
  - Confusion Matrix:
    ```
    [[153360  2009]
     [  1335   750]]
    ```

## 3. Support Vector Machine (SVM)
### Learnings
- Required feature scaling for effective training.
- Training on large datasets was computationally expensive; LinearSVC was used to reduce training time.
- SMOTE (Synthetic Minority Oversampling Technique) was necessary to improve performance on the imbalanced dataset.

### Performance Metrics
- **Accuracy**: 0.5887
- **Precision (class 1)**: 0.0214
- **Recall (class 1)**: 0.6729
- **F1-Score (class 1)**: 0.0415
- **Confusion Matrix**:
  ```
  [[91287 64082]
   [  682  1403]]
  ```

## 4. Random Forest
### Learnings
- Configured with `n_estimators=100`, `max_depth=70`, and `random_state=42` for initial testing.
- Achieved high accuracy and good performance for non-phishing emails but struggled with phishing email precision and recall.
- Hyperparameter tuning with RandomSearchCV slightly improved results, but GridSearchCV was avoided due to high computational cost.

### Performance Metrics (Initial)
- **Accuracy**: 0.98
- **Precision (class 1)**: 0.37
- **Recall (class 1)**: 0.37
- **F1-Score (class 1)**: 0.37
- **Confusion Matrix**:
  ```
  [[154046   1323]
   [  1319    766]]
  ```

### Performance Metrics (RandomSearchCV)
- Configuration:
  ```python
  RandomForestClassifier(
      n_estimators=100,
      min_samples_split=2,
      min_samples_leaf=1,
      max_features='log2',
      max_depth=None,
      bootstrap=False,
      random_state=42
  )
  ```
- **Accuracy**: 0.9852
- **Precision (class 1)**: 0.43
- **Recall (class 1)**: 0.35
- **F1-Score (class 1)**: 0.38
- **Confusion Matrix**:
  ```
  [[154390    979]
   [  1359    726]]
  ```

## 5. K-Nearest Neighbors (KNN)
### Learnings
- Trained quickly, enabling fast experimentation.
- Default `k=5` yielded moderate performance with low precision for phishing emails.
- Testing `k` values from 1 to 20 showed `k=1` as optimal, improving accuracy and F1-score but still struggling with precision.

### Performance Metrics (k=5)
- **Accuracy**: 0.9464
- **Precision (class 1)**: 0.12
- **Recall (class 1)**: 0.47
- **F1-Score (class 1)**: 0.19
- **Confusion Matrix**:
  ```
  [[148049   7320]
   [  1115    970]]
  ```

### Performance Metrics (k=1)
- **Accuracy**: 0.9751
- **Precision (class 1)**: 0.23
- **Recall (class 1)**: 0.38
- **F1-Score (class 1)**: 0.29
- **Confusion Matrix**:
  ```
  [[152731   2638]
   [  1288    797]]
  ```

## Key Observations
- **Imbalanced Data**: All models struggled with low precision for phishing emails (class 1) due to the imbalanced dataset. Techniques like SMOTE, class weighting, or resampling were critical.
- **Trade-offs**: Models like Logistic Regression and SVM achieved high recall but poor precision, while Random Forest and Decision Trees balanced precision and recall better at higher computational costs.
- **Hyperparameter Tuning**: Random Forest and Decision Trees benefited from tuning (e.g., increasing depth or optimizing `k` in KNN), but extensive tuning (e.g., GridSearchCV) was often impractical due to resource constraints.
- **Computational Efficiency**: Decision Trees and KNN trained quickly, while SVM and Random Forest were computationally intensive, especially on large datasets.

## Recommendations
- **Model Selection**: Random Forest with tuned hyperparameters offers the best balance of accuracy, precision, and recall for this task.
- **Data Preprocessing**: Continue using SMOTE or similar techniques to address class imbalance.
- **Future Work**: Explore ensemble methods (e.g., stacking) or deep learning models to further improve phishing detection, and consider feature engineering to enhance model performance.