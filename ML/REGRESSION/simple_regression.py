# FOR SIMPLE REGRESION 

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set
    
    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    
    # Scatter plot for actual vs. predicted values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # 45-degree line
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{name}: Actual vs Predicted")
    plt.show()

# Convert results to DataFrame for comparison
results_df = pd.DataFrame(results).T
print(results_df)

# Save the best model based on RMSE
best_model_name = results_df["RMSE"].idxmin()
best_model = models[best_model_name]
joblib.dump(best_model, "best_regression_model.joblib")
print(f"Best model: {best_model_name} saved as 'best_regression_model.joblib'")