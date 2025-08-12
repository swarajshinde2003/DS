import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================
# 🚀 Example dataset
# ==========================
# X, y = ...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# Define classification models
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting (100)": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting (200)": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost (100)": AdaBoostClassifier(n_estimators=100, random_state=42),
    "AdaBoost (200)": AdaBoostClassifier(n_estimators=200, random_state=42),
    "XGBoost (100)": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "XGBoost (200)": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "LightGBM": LGBMClassifier(n_estimators=200, random_state=42),
    "CatBoost": CatBoostClassifier(n_estimators=200, verbose=0, random_state=42)
}

# ==========================
# Train & evaluate models
# ==========================
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Handle multi-class for ROC-AUC
    try:
        y_prob = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except:
        roc = np.nan  # Some models may not have predict_proba

    # Store results
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc
    }

# ==========================
# Results DataFrame
# ==========================
results_df = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False)
print("\n📊 Evaluation Metrics for All Models:")
print(results_df)

# ==========================
# Save best model
# ==========================
best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, "best_classification_model.joblib")
print(f"\n🏆 Best model: {best_model_name} saved as 'best_classification_model.joblib'")

# ==========================
# Visualization
# ==========================
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y="Accuracy", data=results_df)
plt.xticks(rotation=45, ha='right')
plt.title("Accuracy Comparison of Classification Models")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.show()
