import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("USvideos.csv")

# -----------------------------------
# CREATE TARGET COLUMN
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

# Regression target
y = df['views']

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
# LINEAR REGRESSION MODEL
# -----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# EVALUATION METRICS
# -----------------------------------

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Metrics:\n")

print(f"MSE: {mse}")

print(f"RMSE: {rmse}")

print(f"MAE: {mae}")

print(f"R2 Score: {r2}")

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nTop Features Influencing Views:\n")

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

plt.title("Top Feature Importance - Linear Regression")

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.show()