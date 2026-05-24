import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create additional features
df = pd.read_csv("USvideos.csv")
# Title length
df['title_length'] = df['title'].astype(str).apply(len)

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'])

# Calculate days since publish
latest_date = df['publish_time'].max()

df['days_since_publish'] = (
    latest_date - df['publish_time']
).dt.days

# -----------------------------------
# Histogram of Views
# -----------------------------------

plt.figure(figsize=(10,5))

plt.hist(df['views'], bins=50)

plt.title("Distribution of Views")
plt.xlabel("Views")
plt.ylabel("Frequency")

plt.show()

# -----------------------------------
# Scatter Plot: Likes vs Views
# -----------------------------------

plt.figure(figsize=(10,5))

plt.scatter(df['likes'], df['views'])

plt.title("Likes vs Views")
plt.xlabel("Likes")
plt.ylabel("Views")

plt.show()

# -----------------------------------
# Correlation Matrix
# -----------------------------------

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

# -----------------------------------
# Heatmap
# -----------------------------------

plt.figure(figsize=(10,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix Heatmap")

plt.show() 