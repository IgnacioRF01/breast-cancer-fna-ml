"""Unsupervised model: Gaussian Mixture Model clustering into two groups.

Cluster labels are aligned with the true diagnosis by majority vote, only to
evaluate the clustering. A k-means baseline is included for comparison.
Run from the repository root:  python src/unsupervised_gmm.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from data_loading import load_wdbc

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def align_labels(cluster_labels, y_true):
    """Map each cluster to the most common true label in it (majority vote)."""
    aligned = np.zeros_like(cluster_labels)
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        aligned[mask] = mode(y_true[mask], keepdims=False)[0]
    return aligned


X, y = load_wdbc()
y_true = y.values
X_scaled = StandardScaler().fit_transform(X)

# Gaussian Mixture Model with two components (benign / malignant)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = align_labels(gmm.fit_predict(X_scaled), y_true)

print(f"GMM accuracy: {accuracy_score(y_true, gmm_labels):.4f}")
print("Confusion matrix (aligned clusters vs true labels):\n", confusion_matrix(y_true, gmm_labels))
print("Classification report:\n", classification_report(y_true, gmm_labels, target_names=["Benign", "Malignant"]))

# Baseline: k-means
km_labels = align_labels(KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled), y_true)
print(f"k-means accuracy (baseline): {accuracy_score(y_true, km_labels):.4f}")

# Visualization on the first two standardized features
plt.figure(figsize=(7, 6))
sc = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap="viridis", alpha=0.7)
plt.colorbar(sc, label="Cluster (0 = benign, 1 = malignant)", ticks=[0, 1])
plt.xlabel(f"{X.columns[0]} (standardized)")
plt.ylabel(f"{X.columns[1]} (standardized)")
plt.title("Gaussian Mixture Model clustering of tumors")
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(FIG_DIR, "gmm_clusters.png"), dpi=150)
