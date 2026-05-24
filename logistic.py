import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("USvideos.csv")

# -----------------------------------
# CREATE VIRAL COLUMN
# -----------------------------------

viral_threshold = df['views'].quantile(0.75)

df['viral'] = (df['views'] > viral_threshold).astype(int)

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

# Create title length feature
df['title_length'] = df['title'].astype(str).apply(len)

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'])

# Calculate days since publish
latest_date = df['publish_time'].max()

df['days_since_publish'] = (
    latest_date - df['publish_time']
).dt.days

# -----------------------------------
# SELECT FEATURES
# -----------------------------------

numeric_features = [
    'likes',
    'dislikes',
    'comment_count',
    'title_length',
    'days_since_publish'
]

categorical_features = [
    'channel_title',
    'category_id'
]

# One-hot encoding
encoded_data = pd.get_dummies(
    df[categorical_features],
    drop_first=True
)

# Combine features
X = pd.concat(
    [df[numeric_features], encoded_data],
    axis=1
)

# Classification target
y = df['viral']

# -----------------------------------
# SCALE FEATURES
# -----------------------------------

scaler = StandardScaler()

X[numeric_features] = scaler.fit_transform(
    X[numeric_features]
)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# LOGISTIC REGRESSION MODEL
# -----------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# Prediction probabilities
y_prob = model.predict_proba(X_test)[:,1]

# -----------------------------------
# EVALUATION METRICS
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression Metrics:\n")

print(f"Accuracy: {accuracy}")

print(f"Precision: {precision}")

print(f"Recall: {recall}")

print(f"F1 Score: {f1}")

print(f"ROC-AUC Score: {roc_auc}")

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# -----------------------------------
# ROC CURVE
# -----------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nTop Features Influencing Virality:\n")

print(feature_importance.head(10))

# -----------------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------------

plt.figure(figsize=(12,6))

plt.bar(
    feature_importance['Feature'][:10],
    feature_importance['Coefficient'][:10]
)

plt.xticks(rotation=90)

plt.title("Top Feature Importance - Logistic Regression")

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.show()