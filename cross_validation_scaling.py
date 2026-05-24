import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
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

# Targets
y_regression = df['views']

y_classification = df['viral']

# -----------------------------------
# STANDARD SCALER
# -----------------------------------

standard_scaler = StandardScaler()

X_standard = X.copy()

X_standard[numeric_features] = standard_scaler.fit_transform(
    X_standard[numeric_features]
)

# -----------------------------------
# MINMAX SCALER
# -----------------------------------

minmax_scaler = MinMaxScaler()

X_minmax = X.copy()

X_minmax[numeric_features] = minmax_scaler.fit_transform(
    X_minmax[numeric_features]
)

# -----------------------------------
# LINEAR REGRESSION CROSS VALIDATION
# -----------------------------------

linear_model = LinearRegression()

linear_cv_scores_standard = cross_val_score(
    linear_model,
    X_standard,
    y_regression,
    cv=5,
    scoring='r2'
)

linear_cv_scores_minmax = cross_val_score(
    linear_model,
    X_minmax,
    y_regression,
    cv=5,
    scoring='r2'
)

# -----------------------------------
# LOGISTIC REGRESSION CROSS VALIDATION
# -----------------------------------

logistic_model = LogisticRegression(max_iter=1000)

logistic_cv_scores_standard = cross_val_score(
    logistic_model,
    X_standard,
    y_classification,
    cv=5,
    scoring='accuracy'
)

logistic_cv_scores_minmax = cross_val_score(
    logistic_model,
    X_minmax,
    y_classification,
    cv=5,
    scoring='accuracy'
)

# -----------------------------------
# PRINT RESULTS
# -----------------------------------

print("\nLinear Regression Cross Validation\n")

print("StandardScaler Mean R2 Score:")
print(linear_cv_scores_standard.mean())

print("StandardScaler Variance:")
print(linear_cv_scores_standard.var())

print("\nMinMaxScaler Mean R2 Score:")
print(linear_cv_scores_minmax.mean())

print("MinMaxScaler Variance:")
print(linear_cv_scores_minmax.var())

print("\n-----------------------------------")

print("\nLogistic Regression Cross Validation\n")

print("StandardScaler Mean Accuracy:")
print(logistic_cv_scores_standard.mean())

print("StandardScaler Variance:")
print(logistic_cv_scores_standard.var())

print("\nMinMaxScaler Mean Accuracy:")
print(logistic_cv_scores_minmax.mean())

print("MinMaxScaler Variance:")
print(logistic_cv_scores_minmax.var())

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

logistic_model.fit(
    X_standard,
    y_classification
)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logistic_model.coef_[0]
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

plt.title("Feature Importance - Logistic Regression")

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.show()