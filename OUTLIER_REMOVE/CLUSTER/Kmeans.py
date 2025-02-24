# FOR CLUSTERING ALSO CONSISTS ELBOW AND SHILHOUTEE SCORE 

import pandas as pd

# Set threshold for IQR method
threshold = 1.5  

# Identify numeric columns
num_cols = df.select_dtypes(include=['number']).columns

# Compute IQR
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Compute outlier bounds
upper_bound = Q3 + threshold * IQR  # Only focusing on high outliers

# Find outliers above the upper bound
outlier_mask = (df[num_cols] > upper_bound)  # True for high outliers
outliers_df = df[outlier_mask.any(axis=1)].copy()  # Select only outlier rows

# Compute "outlier severity" (how far each value exceeds the bound)
outliers_df["Outlier_Severity"] = (df[num_cols] - upper_bound).sum(axis=1)

# Sort by severity (most extreme values first)
outliers_df = outliers_df.sort_values(by="Outlier_Severity", ascending=False)

# Select top 10 most extreme outliers
top_10_outliers = outliers_df.head(10)

# Remove these 10 outliers from the original dataset
df_cleaned = df.drop(top_10_outliers.index)

# Print summary
print(f"\n✅ Removed Top 10 Extreme Outliers.")
print(f"📊 Final dataset size: {df_cleaned.shape}")



# TO GET THE SAME CLUSTER COLUMN IN MAIN DF >>>just dont change the sequence

df["Cluster"] = df_pca["Cluster"]