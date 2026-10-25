"""Load the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

Uses data/data.csv if present (the CSV version used in the original project:
id, diagnosis, 30 features). Otherwise falls back to the identical copy bundled
with scikit-learn, so the scripts run with no download.
Returns X (569 x 30 DataFrame) and y (Series: 0 = benign, 1 = malignant).
"""
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")


def load_wdbc(csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        if data.shape[1] > 32:                       # drop trailing empty columns
            data = data.iloc[:, :32]
        data = data.drop(columns=["id", "ID"], errors="ignore")
        y = data["diagnosis"].map({"B": 0, "M": 1})
        X = data.drop(columns="diagnosis")
        return X, y
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(1 - ds.target, name="diagnosis")  # sklearn: 0 = malignant -> flip
    return X, y
