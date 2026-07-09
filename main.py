
# importinggg
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier



# Create a folder to save all generated figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# Step 1: Loading Dataset
df = pd.read_csv("Loan Approval Dataset.csv")
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


# Step 2: Handling Missing Values
if df.isnull().values.any():
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
else:
    df_imputed=df


# Step 3: Categorical Variables Encoding
label_encoders = {}
for column in df_imputed.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_imputed[column] = le.fit_transform(df_imputed[column])
    label_encoders[column] = le



# Step 4: Plotting Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.show()



# Step 5: Splitting Train-Test Split by 70/30 ratio
X = df_imputed.drop("loan_status", axis=1)
y = df_imputed["loan_status"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)



# Step 6: Define Models

models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(max_iter=1000)
}



# Step 7: Apply Scaling and Train Models

results = {}

for name, model in models.items():
    if name == "KNN":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    if hasattr(model, "predict_proba"):
          y_prob = model.predict_proba(X_test)[:, 1]
    else:
         y_prob= y_pred

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1]
    }


# Step 8: Summary Table
summary_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "ROC AUC": res["roc_auc"]
    }
    for name, res in results.items()
])
print("\n                                   Model Evaluation Summary")
print(summary_df)



# Step 9: Accuracy Bar-Chart Representation
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=summary_df)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_model_accuracy.png"), dpi=150)
plt.show()



# Step 10: Precision & Recall Bar-Chart Representation
summary_melted = pd.melt(summary_df, id_vars="Model", value_vars=["Precision", "Recall"])
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="value", hue="variable", data=summary_melted)
plt.title("Precision vs Recall by Model")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_precision_recall.png"), dpi=150)
plt.show()

# Step 11: Confusion Matrices Diagram
for name, res in results.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES_DIR, f"05_confusion_matrix_{safe_name}.png"), dpi=150)
    plt.show()



# Step 12: ROC Curve Comparison Graph
plt.figure(figsize=(10, 8))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_roc_curve.png"), dpi=150)
plt.show()