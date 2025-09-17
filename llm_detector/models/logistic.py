"""Thin wrapper around scikit-learn logistic regression."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from llm_detector.training.dataset import FeatureDataset


@dataclass(slots=True)
class TrainingResult:
    """Summary returned after fitting a logistic model."""

    train_accuracy: float
    metrics: dict[str, float]
    feature_names: list[str]


class LogisticRegressionModel:
    """Wrapper that manages scaling, training, and persistence."""

    def __init__(
        self,
        *,
        class_weight: str | dict[int, float] | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_names: list[str] | None = None
        self._fitted = False

    @staticmethod
    def _as_array(matrix: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError("feature matrix must be two-dimensional")
        return arr

    @staticmethod
    def _as_vector(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.ndim != 1:
            raise ValueError("feature vector must be one-dimensional")
        return arr

    def fit(self, dataset: FeatureDataset) -> TrainingResult:
        if not dataset.matrix:
            raise ValueError("training dataset is empty")

        X = self._as_array(dataset.matrix)
        y = np.asarray(dataset.labels, dtype=int)
        if X.shape[0] != y.shape[0]:
            raise ValueError("feature matrix and labels have mismatched rows")

        self.feature_names = list(dataset.feature_names)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True

        preds = self._model.predict(X_scaled)
        accuracy = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
        metrics = {
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
        }
        return TrainingResult(
            train_accuracy=float(accuracy), metrics=metrics, feature_names=self.feature_names
        )

    def predict_proba(self, features: Sequence[float]) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("model must be trained before prediction")
        vector = self._as_vector(features)
        if self.feature_names is not None and len(vector) != len(self.feature_names):
            raise ValueError("feature vector length does not match training features")
        matrix = vector.reshape(1, -1)
        scaled = self._scaler.transform(matrix)
        probs = self._model.predict_proba(scaled)[0]
        return float(probs[0]), float(probs[1])

    def predict(self, features: Sequence[float]) -> int:
        _, p_llm = self.predict_proba(features)
        return int(p_llm >= 0.5)

    def save(self, path: Path | str) -> None:
        if not self._fitted or self.feature_names is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self._scaler,
            "model": self._model,
            "feature_names": self.feature_names,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> LogisticRegressionModel:
        payload = joblib.load(path)
        obj = cls(
            class_weight=payload.get("class_weight", "balanced"),
            max_iter=payload.get("max_iter", 1000),
            random_state=payload.get("random_state", 42),
        )
        obj._scaler = payload["scaler"]
        obj._model = payload["model"]
        obj.feature_names = payload.get("feature_names")
        obj._fitted = True
        return obj

    def evaluate(self, dataset: FeatureDataset) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("model must be trained before evaluation")
        if not dataset.matrix:
            raise ValueError("evaluation dataset is empty")

        X = self._as_array(dataset.matrix)
        y = np.asarray(dataset.labels, dtype=int)
        scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(scaled)
        preds = (probs[:, 1] >= 0.5).astype(int)

        accuracy = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


__all__ = ["LogisticRegressionModel", "TrainingResult"]
