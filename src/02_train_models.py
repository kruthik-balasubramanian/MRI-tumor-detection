"""
Step 2: Model Training, Evaluation & Selection
Brain Tumor Classification from MRI — using extracted HOG + GLCM features
4-class problem: no_tumor, glioma_tumor, meningioma_tumor, pituitary_tumor
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42

# ---------------- Load extracted features ----------------
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train_raw = pd.read_csv("data/y_train.csv")["label"].values
y_test_raw = pd.read_csv("data/y_test.csv")["label"].values

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
class_names = le.classes_
print("Classes:", list(class_names))

# ---------------- Scale + reduce dimensionality ----------------
# 1812 raw features on ~2870 samples risks overfitting for some models,
# and slows SVM/KNN down a lot -- PCA keeps the signal, drops the noise.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components (95% variance retained)")

np.save("data/X_train_pca_cache.npy", X_train_pca)
np.save("data/X_test_pca_cache.npy", X_test_pca)
np.save("data/y_train_cache.npy", y_train)
np.save("data/y_test_cache.npy", y_test)

# ---------------- Models ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Support Vector Machine": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Macro": f1_score(y_test, y_pred, average="macro"),
    }
    results.append(metrics)
    print(f"\n{'='*55}\n{name}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=class_names))

results_df = pd.DataFrame(results).sort_values("F1-Macro", ascending=False)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\n\nMODEL COMPARISON (sorted by F1-Macro):\n", results_df.to_string(index=False))
print(f"\nBest baseline model: {results_df.iloc[0]['Model']} -> run src/02b_tune_best_model.py next")

# ---------------- Plots that don't depend on the tuned model ----------------
plt.figure(figsize=(8, 6))
data_bar = results_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Macro"])
sns.barplot(data=data_bar, x="Model", y="value", hue="variable")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, 1)
plt.title("Model Comparison — Brain Tumor MRI Classification")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Variance Retained")
plt.axhline(0.95, color="red", linestyle="--", alpha=0.6, label="95% threshold")
plt.legend()
plt.tight_layout()
plt.savefig("results/pca_variance.png", dpi=150)
plt.close()

# ---------------- Save preprocessing artifacts ----------------
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("\nSaved scaler, PCA transform, and label encoder to models/")
print("Saved plots -> results/")
