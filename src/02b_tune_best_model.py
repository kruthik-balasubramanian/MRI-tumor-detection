"""
Step 2b: Hyperparameter tuning for the best model (XGBoost), using cached
PCA-transformed features from 02_train_models.py so we don't repeat the
expensive feature-scaling/PCA step on every run.
"""
import numpy as np
import pandas as pd
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

X_train_pca = np.load("data/X_train_pca_cache.npy")
X_test_pca = np.load("data/X_test_pca_cache.npy")
y_train = np.load("data/y_train_cache.npy")
y_test = np.load("data/y_test_cache.npy")

le = joblib.load("models/label_encoder.joblib")
class_names = le.classes_

# Single stratified train/validation split for tuning (not full k-fold CV) --
# on this single-core machine, a full grid search with cross-validation is
# too slow (each XGBoost fit takes ~6-7s, and CV multiplies that by k-folds
# x n_combinations). A held-out validation split gives a reasonable signal
# for picking hyperparameters at a fraction of the compute cost.
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_pca, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

candidates = [
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05},
]

best_score, best_params = -1, None
for params in candidates:
    m = XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, **params)
    m.fit(X_tr, y_tr)
    score = f1_score(y_val, m.predict(X_val), average="macro")
    print(f"  params={params} -> val F1-macro={score:.4f}")
    if score > best_score:
        best_score, best_params = score, params

print("Best params:", best_params)

# Refit best config on the FULL training set (not just the tuning split)
best_model = XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, **best_params)
best_model.fit(X_train_pca, y_train)

y_pred_final = best_model.predict(X_test_pca)
final_metrics = {
    "model": "XGBoost",
    "Accuracy": accuracy_score(y_test, y_pred_final),
    "F1_Macro": f1_score(y_test, y_pred_final, average="macro"),
    "best_params": best_params,
}
print("\nFinal tuned model metrics:", final_metrics)
with open("results/final_model_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print("\nFinal classification report:")
print(classification_report(y_test, y_pred_final, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — Tuned XGBoost (Final Model)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

joblib.dump(best_model, "models/best_model.joblib")
print("\nSaved tuned model -> models/best_model.joblib")
