import pandas as pd

# Set threshold for IQR method
threshold = 1.5  

# Store original data length
original_length = len(df)

# Identify numeric columns
num_cols = df.select_dtypes(include=['number']).columns

# Calculate IQR bounds for all numeric columns
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Compute lower and upper bounds
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR

# Create a mask for rows that are outliers in ANY column
outlier_mask = (df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)
outliers_per_row = outlier_mask.any(axis=1)  # Find rows with any outlier

# Count total outliers before removal
total_outliers = outliers_per_row.sum()
print(f"📉 Total rows containing outliers: {total_outliers}")

# Remove all rows with at least one outlier
df = df[~outliers_per_row]

# Print summary
removed_outliers = original_length - len(df)
print(f"\n✅ Total Outliers Removed: {removed_outliers}")
print(f"📊 Final dataset size: {df.shape}")