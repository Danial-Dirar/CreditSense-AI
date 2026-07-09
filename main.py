
# importinggg
import os
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Create a folder to save all generated figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# Step 1: Loading Dataset
df = pd.read_csv("Loan Approval Dataset.csv")

# Loan_ID is just a unique identifier (LP001002, ...), not a real feature,
# so we drop it to keep it out of the models.
df = df.drop(columns=["Loan_ID"])

# plotting for checkk
plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", data=df, order=["N", "Y"])
plt.title("Class Distribution (Before Preprocessing)")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks([0, 1], ["Rejected (0)", "Approved (1)"])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_class_distribution.png"), dpi=150)
plt.show()


# Step 2: Features / Target
# Target: 1 = Approved (Y), 0 = Rejected (N)
y = (df["loan_status"] == "Y").astype(int)
X = df.drop(columns=["loan_status"])

# Split columns by type so each gets the right preprocessing
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()


# Step 3: Correlation Heatmap
# For the heatmap we need everything numeric, so we encode categoricals just
# for visualisation (this encoded frame is NOT used for training).
df_encoded = df.copy()
for col in categorical_features + ["loan_status"]:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.show()


# Step 4: Preprocessing definition
# - Numeric  : impute missing with the MEDIAN, then scale.
# - Category : impute missing with the MOST FREQUENT value, then One-Hot encode.
# One-Hot avoids the false "ordering" that LabelEncoder imposes on categories.
def build_preprocessor(numeric_scaler):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", numeric_scaler),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])


# Step 5: Define Models
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


# Step 6: Train & Evaluate with 5-fold Cross-Validation
# Instead of one lucky train/test split, we use out-of-fold predictions so that
# every row is predicted by a model that never saw it during training. This
# gives far more stable, honest estimates on a small (614-row) dataset.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    # KNN is distance-based -> MinMax; the rest -> StandardScaler.
    scaler = MinMaxScaler() if name == "KNN" else StandardScaler()
    pipe = Pipeline([
        ("preprocess", build_preprocessor(scaler)),
        ("model", model),
    ])

    # Out-of-fold probabilities, then threshold at 0.5 for the class prediction.
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, y_prob)
    results[name] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }


# Step 7: Summary Table
summary_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1": res["f1"],
        "ROC AUC": res["roc_auc"],
    }
    for name, res in results.items()
]).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

baseline = max(y.mean(), 1 - y.mean())
print("\n                          Model Evaluation Summary (5-fold CV)")
print(summary_df.to_string(index=False))
print(f"\nBaseline accuracy (always predict the majority class): {baseline:.3f}")


# Step 8: Accuracy Bar-Chart Representation
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=summary_df)
plt.title("Model Accuracy Comparison")
plt.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline:.2f})")
plt.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_model_accuracy.png"), dpi=150)
plt.show()


# Step 9: Precision / Recall / F1 Bar-Chart Representation
summary_melted = pd.melt(summary_df, id_vars="Model",
                         value_vars=["Precision", "Recall", "F1"])
plt.figure(figsize=(9, 5))
sns.barplot(x="Model", y="value", hue="variable", data=summary_melted)
plt.title("Precision / Recall / F1 by Model")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_precision_recall.png"), dpi=150)
plt.show()


# Step 10: Confusion Matrices Diagram
for name, res in results.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES_DIR, f"05_confusion_matrix_{safe_name}.png"), dpi=150)
    plt.show()


# Step 11: ROC Curve Comparison Graph
plt.figure(figsize=(10, 8))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_roc_curve.png"), dpi=150)
plt.show()
