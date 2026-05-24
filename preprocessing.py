from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# -----------------------------------
# TASK 4
# Feature Preprocessing
# -----------------------------------
df = pd.read_csv("USvideos.csv")
# Select numeric features
numeric_features = [
    'likes',
    'dislikes',
    'comment_count',
    'title_length',
    'days_since_publish'
]

# Select categorical features
categorical_features = [
    'channel_title',
    'category_id'
]

# One-hot encoding
encoded_data = pd.get_dummies(
    df[categorical_features],
    drop_first=True
)

# Combine numeric + encoded categorical features
X = pd.concat(
    [df[numeric_features], encoded_data],
    axis=1
)

# Regression target
y_regression = df['views']

# Classification target
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