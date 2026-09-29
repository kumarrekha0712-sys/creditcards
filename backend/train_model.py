"""
train_model.py
----------------
Trains a Logistic Regression fraud-detection model and caches everything
the API needs (model + test-set probabilities) so the FastAPI backend
never has to retrain when the user moves the threshold slider.

Dataset
-------
The real Kaggle "Credit Card Fraud Detection" dataset (creditcard.csv,
~284,807 rows, features V1-V28 + Time + Amount + Class) is not bundled
with this project because of its size/license. This script looks for it
at ../dataset/creditcard.csv first.

If that file is not found, it automatically GENERATES a synthetic dataset
that mimics the real one's shape and class imbalance (Time, V1-V28, Amount,
Class, ~0.17% fraud rate) using sklearn.datasets.make_classification, so
the whole project is runnable out of the box. Swap in the real CSV any
time -- no other code changes needed.

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import joblib

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BACKEND_DIR, "..", "dataset", "creditcard.csv")
MODEL_PATH = os.path.join(BACKEND_DIR, "model.pkl")
CACHE_PATH = os.path.join(BACKEND_DIR, "eval_cache.pkl")

RANDOM_STATE = 42


def generate_synthetic_dataset(n_samples=20000, fraud_ratio=0.0172):
    """
    Creates a synthetic dataset shaped like the real creditcard.csv:
    Time, V1..V28 (PCA-like anonymized features), Amount, Class.
    Fraud ratio defaults to ~1.7% (heavier than the real 0.17% so the
    demo trains fast and still shows meaningful imbalance behaviour).
    """
    n_features = 28  # V1..V28
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=6,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.001,
        class_sep=1.1,
        random_state=RANDOM_STATE,
    )

    rng = np.random.RandomState(RANDOM_STATE)
    time_col = np.sort(rng.randint(0, 172792, size=n_samples)).astype(float)
    # Fraudulent amounts skew slightly lower on average, like real data
    amount_col = np.where(
        y == 1,
        np.abs(rng.normal(loc=80, scale=100, size=n_samples)),
        np.abs(rng.normal(loc=90, scale=180, size=n_samples)),
    )

    columns = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "Time", time_col)
    df["Amount"] = amount_col
    df["Class"] = y

    return df


def load_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading real dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        if "Class" not in df.columns:
            raise ValueError("Dataset must contain a 'Class' target column (0=legit, 1=fraud).")
        return df
    else:
        print("Real dataset not found at dataset/creditcard.csv.")
        print("Generating a synthetic credit-card-fraud-like dataset instead...")
        df = generate_synthetic_dataset()
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Synthetic dataset saved to {DATASET_PATH} ({len(df)} rows).")
        return df


def main():
    df = load_dataset()

    print(f"Dataset shape: {df.shape}")
    fraud_count = int(df["Class"].sum())
    print(f"Fraud cases: {fraud_count} ({fraud_count / len(df) * 100:.3f}%)")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_columns = list(X.columns)

    # Stratified split so both train and test keep the same fraud ratio.
    # This is done BEFORE any resampling/scaling is fit, to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # Scale using statistics from the TRAINING set only (no leakage).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance via class_weight="balanced" instead of
    # resampling the test set (resampling test data would itself be a
    # form of leakage/evaluation bias). This reweights the loss function
    # during training only.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # Predicted fraud probabilities on the untouched test set.
    test_probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Save model + scaler together (scaler is needed for live /predict calls).
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": feature_columns},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")

    # Cache test-set probabilities + true labels so the API can recompute
    # metrics for any threshold WITHOUT retraining or re-predicting.
    joblib.dump(
        {
            "y_true": y_test.to_numpy(),
            "y_proba": test_probabilities,
            "X_test_raw": X_test.reset_index(drop=True),
        },
        CACHE_PATH,
    )
    print(f"Evaluation cache saved to {CACHE_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
