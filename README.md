# YouTube Analytics Machine Learning Project

## Overview

This project analyzes YouTube trending video data and applies Machine Learning techniques to:

- Predict video views using Linear Regression
- Classify videos as viral or non-viral using Logistic Regression
- Perform hypothesis testing on engagement metrics
- Compare scaling techniques and evaluate model performance

The project uses the US YouTube Trending Videos dataset.

---

# Objectives

- Explore YouTube engagement metrics
- Understand relationships between likes, comments, dislikes, and views
- Predict video views using regression
- Predict virality using classification
- Apply preprocessing and feature engineering
- Evaluate machine learning models
- Generate business insights for content strategy

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

# Dataset

Dataset Used:
- USvideos.csv

Features include:
- views
- likes
- dislikes
- comment_count
- publish_time
- category_id
- channel_title

---

# Project Structure

```bash
Youtube-Analytics-Machine-Learning/
│
├── youtube_analytics_ml.py
├── linear_regression.py
├── logistic_regression.py
├── cross_validation_scaling.py
├── final_report.py
├── USvideos.csv
├── requirements.txt
└── README.md
```

---

# Tasks Performed

## Task 1: Dataset Loading & Target Creation
- Loaded dataset using Pandas
- Inspected data using:
  - head()
  - info()
  - describe()
- Created viral target column

---

## Task 2: Data Visualization & Correlation Analysis
- Histogram of views
- Scatterplot of likes vs views
- Correlation matrix
- Heatmap visualization

---

## Task 3: Hypothesis Testing
- Performed t-tests
- Calculated p-values
- Tested relationship between:
  - likes
  - dislikes
  - comment_count
  and views

---

## Task 4: Feature Preprocessing
- One-hot encoding
- Feature scaling using StandardScaler
- Train-test split
- Feature engineering:
  - title_length
  - days_since_publish

---

## Task 5: Linear Regression
Evaluation Metrics:
- MSE
- RMSE
- MAE
- R² Score

Visualized feature importance.

---

## Task 6: Logistic Regression
Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Generated:
- Confusion Matrix
- ROC Curve
- Feature importance plot

---

## Task 7: Cross Validation & Scaling Comparison
- 5-Fold Cross Validation
- Compared:
  - StandardScaler
  - MinMaxScaler
- Analyzed feature importance

---

## Task 8: Final Report & Business Insights
Generated conclusions and recommendations for:
- Improving video engagement
- Increasing virality
- Understanding audience interaction

---

# Key Insights

- Likes strongly influence video views.
- Comment count positively impacts virality.
- High engagement videos are more likely to become viral.
- Audience interaction is an important factor in YouTube growth.

---

# How to Run

## Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## Run Files

### Main Analysis
```bash
py youtube_analytics_ml.py
```

### Linear Regression
```bash
py linear_regression.py
```

### Logistic Regression
```bash
py logistic_regression.py
```

### Cross Validation
```bash
py cross_validation_scaling.py
```

### Final Report
```bash
py final_report.py
```

---

# Machine Learning Concepts Used

- Regression
- Classification
- Feature Engineering
- Hypothesis Testing
- Cross Validation
- Feature Scaling
- Data Visualization
- Model Evaluation

---

# Future Improvements

- Add Deep Learning models
- Deploy models using FastAPI
- Build interactive dashboard
- Use real-time YouTube API data
- Add NLP-based title analysis

---

# Author

Karan Kumar

AI Enthusiast | Machine Learning | GenAI | Drone Technology