"""Supervised model: standardization + PCA (10 components) + logistic regression.

Run from the repository root:  python src/supervised_pca_logreg.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loading import load_wdbc

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

X, y = load_wdbc()

# 80 % training / 20 % testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize (fitted on the training set only, then applied to the test set)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA: keep 10 principal components (fitted on the training set only)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("Total explained variance (10 PCs):", round(pca.explained_variance_ratio_.sum(), 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign (B)", "Malignant (M)"], yticklabels=["Benign (B)", "Malignant (M)"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix: PCA + logistic regression (test set)")
plt.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_supervised.png"), dpi=150)
