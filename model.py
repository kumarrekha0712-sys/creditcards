"""
model.py
--------
Loads the trained model/scaler and the cached test-set probabilities,
and computes confusion-matrix-based metrics for any given threshold
WITHOUT retraining or re-running predictions.
"""

import os
import numpy as np
import joblib

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "model.pkl")
CACHE_PATH = os.path.join(BACKEND_DIR, "eval_cache.pkl")


class FraudModelUnavailable(Exception):
    """Raised when model.pkl / eval_cache.pkl are missing or unreadable."""
    pass


class FraudModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.y_true = None
        self.y_proba = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FraudModelUnavailable(
                f"model.pkl not found at {MODEL_PATH}. Run 'python train_model.py' first."
            )
        if not os.path.exists(CACHE_PATH):
            raise FraudModelUnavailable(
                f"eval_cache.pkl not found at {CACHE_PATH}. Run 'python train_model.py' first."
            )

        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.feature_columns = bundle["feature_columns"]

            cache = joblib.load(CACHE_PATH)
            self.y_true = np.asarray(cache["y_true"])
            self.y_proba = np.asarray(cache["y_proba"])
        except Exception as exc:
            raise FraudModelUnavailable(f"Failed to load model artifacts: {exc}") from exc

    def is_ready(self):
        return self.model is not None and self.y_proba is not None

    def predict_single(self, features: dict, threshold: float):
        """Predict fraud probability + label for a single transaction dict."""
        if not self.is_ready():
            raise FraudModelUnavailable("Model is not loaded.")

        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = np.array([[features[c] for c in self.feature_columns]], dtype=float)
        row_scaled = self.scaler.transform(row)
        proba = float(self.model.predict_proba(row_scaled)[0, 1])
        prediction = 1 if proba >= threshold else 0
        return {"fraud_probability": proba, "prediction": prediction, "threshold": threshold}

    def evaluate_at_threshold(self, threshold: float):
        """
        Recomputes predictions + all metrics for the cached test-set
        probabilities at the given threshold. No retraining involved.
        """
        if not self.is_ready():
            raise FraudModelUnavailable("Model is not loaded.")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")

        y_true = self.y_true
        y_pred = (self.y_proba >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "threshold": round(float(threshold), 4),
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "total_test_transactions": total,
            "total_actual_fraud": int(np.sum(y_true == 1)),
        }


# Singleton instance used by main.py. Loaded once at startup; if it fails,
# main.py surfaces a clear error on every endpoint instead of crashing.
try:
    fraud_model = FraudModel()
    MODEL_LOAD_ERROR = None
except FraudModelUnavailable as e:
    fraud_model = None
    MODEL_LOAD_ERROR = str(e)
