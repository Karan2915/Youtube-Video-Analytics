import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# -----------------------------------
# TASK 1
# Load Dataset & Define Targets
# -----------------------------------

# Load dataset
df = pd.read_csv("USvideos.csv")

# Display dataset shape
print("Dataset Shape:")
print(df.shape)

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nDataset Statistics:")
print(df.describe())

# Create viral column
viral_threshold = df['views'].quantile(0.75)

df['viral'] = (df['views'] > viral_threshold).astype(int)

print("\nViral Column Created:")
print(df[['views', 'viral']].head())

# -----------------------------------
# TASK 2
# Explore Distributions & Relationships
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

# Histogram of views
plt.figure(figsize=(10,5))

plt.hist(df['views'], bins=50)

plt.title("Distribution of Views")
plt.xlabel("Views")
plt.ylabel("Frequency")

plt.show()

# Scatter plot likes vs views
plt.figure(figsize=(10,5))

plt.scatter(df['likes'], df['views'])

plt.title("Likes vs Views")
plt.xlabel("Likes")
plt.ylabel("Views")

plt.show()

# Correlation matrix
numeric_features = [
    'views',
    'likes',
    'dislikes',
    'comment_count',
    'title_length',
    'days_since_publish'
]

corr_matrix = df[numeric_features].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix Heatmap")

plt.show()

# -----------------------------------
# TASK 3
# Hypothesis Testing
# -----------------------------------

median_views = df['views'].median()

high_views = df[df['views'] > median_views]

low_views = df[df['views'] <= median_views]

features = ['likes', 'dislikes', 'comment_count']

print("\nHypothesis Testing Results:\n")

for feature in features:

    t_stat, p_value = ttest_ind(
        high_views[feature],
        low_views[feature],
        equal_var=False
    )

    print(f"{feature}")

    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("Result: Significant relationship with views")
    else:
        print("Result: No significant relationship with views")

    print("-----------------------------------")

# -----------------------------------
# TASK 4
# Feature Preprocessing
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

# Scale numeric features
scaler = StandardScaler()

X[numeric_features] = scaler.fit_transform(
    X[numeric_features]
)

print("\nScaled Numeric Features:")
print(X[numeric_features].head())

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_regression,
    test_size=0.2,
    random_state=42
)

# Train-test split for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_classification,
    test_size=0.2,
    random_state=42
)

print("\nRegression Training Shape:")
print(X_train_reg.shape)

print("\nRegression Testing Shape:")
print(X_test_reg.shape)

print("\nClassification Training Shape:")
print(X_train_clf.shape)

print("\nClassification Testing Shape:")
print(X_test_clf.shape)

# -----------------------------------
# TASK 5
# Linear Regression
# -----------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = linear_model.predict(X_test_reg)

# Evaluation Metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test_reg, y_pred_reg)

r2 = r2_score(y_test_reg, y_pred_reg)

print("\nLinear Regression Metrics:")

print(f"MSE: {mse}")

print(f"RMSE: {rmse}")

print(f"MAE: {mae}")

print(f"R2 Score: {r2}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_model.coef_
})

feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

# Plot feature importance
plt.figure(figsize=(12,6))

plt.bar(
    feature_importance['Feature'][:10],
    feature_importance['Coefficient'][:10]
)

plt.xticks(rotation=90)

plt.title("Top Feature Importance - Linear Regression")

plt.show()

# -----------------------------------
# TASK 6
# Logistic Regression
# -----------------------------------

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train_clf, y_train_clf)

# Predictions
y_pred_clf = logistic_model.predict(X_test_clf)

# Prediction probabilities
y_prob_clf = logistic_model.predict_proba(X_test_clf)[:,1]

# Evaluation Metrics
print("\nClassification Report:")

print(classification_report(y_test_clf, y_pred_clf))

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)

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

# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test_clf,
    y_prob_clf
)

roc_auc = roc_auc_score(
    y_test_clf,
    y_prob_clf
)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print(f"\nROC-AUC Score: {roc_auc}")