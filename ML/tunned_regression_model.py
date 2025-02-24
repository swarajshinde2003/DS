# HYPERPARAMETER TUNNED REGRESSION MODELS 
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define hyperparameter grids
param_grids = {
    "Ridge Regression": {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    "Lasso Regression": {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    "ElasticNet Regression": {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10, 15]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10, 15]
    },
    "Support Vector Regressor": {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.2, 0.5],
        "kernel": ["linear", "rbf"]
    },
    "Extra Trees Regressor": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }
}

# Define models
models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    "Support Vector Regressor": SVR(),
    "Extra Trees Regressor": ExtraTreesRegressor(random_state=42)
}

# Dictionary to store best models and results
best_models = {}
results = {}

# Perform hyperparameter tuning using GridSearchCV
for name, model in tqdm(models.items(), desc="Tuning Models"):
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    best_models[name] = best_model
    results[name] = {"Best Params": best_params, "RMSE": rmse, "MAE": mae, "R²": r2}

    # Scatter plot for actual vs. predicted values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{name}: Actual vs Predicted")
    plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Save results to CSV for review
results_df.to_csv("model_performance.csv")

print("\nModel Performance Summary:")
print(results_df)

# Visualize RMSE comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df["RMSE"], palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Model RMSE Comparison")
plt.show()

# Save the best model based on RMSE
best_model_name = results_df["RMSE"].idxmin()
best_model = best_models[best_model_name]
joblib.dump(best_model, "best_regression_model_tuned.joblib")
print(f"Best tuned model: {best_model_name} saved as 'best_regression_model_tuned.joblib'")
