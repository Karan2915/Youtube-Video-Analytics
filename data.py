import pandas as pd

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

# Display viral column
print("\nViral Column Created:")
print(df[['views', 'viral']].head())