from scipy.stats import ttest_ind
import pandas as pd
# -----------------------------------
# TASK 3
# Hypothesis Testing
# -----------------------------------
df = pd.read_csv("USvideos.csv")
# Split dataset into high-view and low-view groups

median_views = df['views'].median()

high_views = df[df['views'] > median_views]

low_views = df[df['views'] <= median_views]

# Features to test
features = ['likes', 'dislikes', 'comment_count']

print("\nHypothesis Testing Results:\n")

for feature in features:

    # Perform t-test
    t_stat, p_value = ttest_ind(
        high_views[feature],
        low_views[feature],
        equal_var=False
    )

    print(f"{feature}")

    print(f"P-value: {p_value}")

    # Interpretation
    if p_value < 0.05:
        print("Result: Significant relationship with views")
    else:
        print("Result: No significant relationship with views")

    print("-----------------------------------")