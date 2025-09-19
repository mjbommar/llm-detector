"""Thin wrapper around scikit-learn logistic regression."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime environments may omit numpy.
    np = None  # type: ignore[assignment]

SERIALIZATION_FORMAT = "llm-detector/logistic-regression"
SERIALIZATION_VERSION = 1


def _ensure_numpy() -> Any:
    if np is None:
        msg = "numpy is required for logistic regression training"
        raise RuntimeError(msg)
    return np


def _compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if accuracy_score is not None:  # pragma: no branch - simple optional dependency
        return float(accuracy_score(y_true, y_pred))

    true_list = list(y_true)
    pred_list = list(y_pred)
    total = len(true_list)
    if total == 0:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(true_list, pred_list, strict=False))
    return float(correct / total)


def _compute_precision_recall_f1(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[float, float, float]:
    if precision_recall_fscore_support is not None:  # pragma: no branch
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        return float(precision), float(recall), float(f1)

    true_list = list(y_true)
    pred_list = list(y_pred)
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for truth, pred in zip(true_list, pred_list, strict=False):
        if truth == 1 and pred == 1:
            true_positive += 1.0
        elif truth == 0 and pred == 1:
            false_positive += 1.0
        elif truth == 1 and pred == 0:
            false_negative += 1.0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

try:  # Training environments require scikit-learn, runtime may not.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - exercised in minimal runtime envs.
    LogisticRegression = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

try:  # Metrics are optional; we fall back to a manual implementation if absent.
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:  # pragma: no cover - exercised in minimal runtime envs.
    accuracy_score = None  # type: ignore[assignment]
    precision_recall_fscore_support = None  # type: ignore[assignment]

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

        if StandardScaler is None or LogisticRegression is None:  # pragma: no cover
            msg = "scikit-learn is required to train LogisticRegressionModel"
            raise RuntimeError(msg)

        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_names: list[str] | None = None
        self._fitted = False
        self._scaler_mean: list[float] | None = None
        self._scaler_scale: list[float] | None = None
        self._coefficients: list[float] | None = None
        self._intercept: float | None = None
        self._classes: list[int] | None = None

    @staticmethod
    def _as_array(matrix: Sequence[Sequence[float]]) -> Any:
        np_mod = _ensure_numpy()
        arr = np_mod.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError("feature matrix must be two-dimensional")
        return arr

    @staticmethod
    def _as_vector(vector: Sequence[float]) -> Any:
        if np is not None:
            arr = np.asarray(vector, dtype=float)
            if arr.ndim != 1:
                raise ValueError("feature vector must be one-dimensional")
            return arr

        values = [float(val) for val in vector]
        if any(isinstance(val, Sequence) and not isinstance(val, (str, bytes)) for val in vector):
            raise ValueError("feature vector must be one-dimensional")
        return values

    def _scale_vector(self, vector: Any) -> list[float] | Any:
        if self._scaler is not None:
            np_mod = _ensure_numpy()
            arr = np_mod.asarray(vector, dtype=float).reshape(1, -1)
            return self._scaler.transform(arr)[0]
        if self._scaler_mean is None or self._scaler_scale is None:
            raise RuntimeError("model is missing scaler parameters")
        values = [float(val) for val in vector]
        if len(values) != len(self._scaler_mean):
            raise RuntimeError("feature vector length does not match scaler parameters")
        scaled: list[float] = []
        for value, mean, scale in zip(values, self._scaler_mean, self._scaler_scale, strict=False):
            denom = 1.0 if scale == 0.0 else scale
            scaled.append((value - mean) / denom)
        return scaled

    def _predict_from_scaled(self, scaled: Any) -> tuple[float, float]:
        if self._model is not None:
            np_mod = _ensure_numpy()
            arr = np_mod.asarray(scaled, dtype=float).reshape(1, -1)
            probs = self._model.predict_proba(arr)[0]
            return float(probs[0]), float(probs[1])

        if self._coefficients is None or self._intercept is None or self._classes is None:
            raise RuntimeError("model is missing coefficients for inference")
        if len(self._classes) != 2 or sorted(self._classes) != [0, 1]:
            raise RuntimeError(f"unsupported class labels: {self._classes!r}")

        coefficients = [float(val) for val in self._coefficients]
        scaled_values = [float(val) for val in scaled]
        if len(coefficients) != len(scaled_values):
            raise RuntimeError("feature vector length does not match coefficient vector")

        linear_term = self._intercept + sum(c * x for c, x in zip(coefficients, scaled_values, strict=False))
        if linear_term >= 0:
            exp_term = math.exp(-linear_term)
            p_positive = 1.0 / (1.0 + exp_term)
        else:
            exp_term = math.exp(linear_term)
            p_positive = exp_term / (1.0 + exp_term)

        class0, class1 = self._classes
        # `p_positive` corresponds to the second class seen during training.
        probabilities = {class0: 1.0 - p_positive, class1: p_positive}
        try:
            return float(probabilities[0]), float(probabilities[1])
        except KeyError as err:  # pragma: no cover - defensive guard for unexpected class ids.
            raise RuntimeError("serialized model must include classes {0, 1}") from err

    def fit(self, dataset: FeatureDataset) -> TrainingResult:
        if not dataset.matrix:
            raise ValueError("training dataset is empty")

        X = self._as_array(dataset.matrix)
        np_mod = _ensure_numpy()
        y = np_mod.asarray(dataset.labels, dtype=int)
        if X.shape[0] != y.shape[0]:
            raise ValueError("feature matrix and labels have mismatched rows")

        self.feature_names = list(dataset.feature_names)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._scaler_mean = self._scaler.mean_.astype(float).tolist()
        self._scaler_scale = self._scaler.scale_.astype(float).tolist()
        self._coefficients = self._model.coef_.astype(float).ravel().tolist()
        self._intercept = float(self._model.intercept_.ravel()[0])
        self._classes = [int(cls) for cls in self._model.classes_]
        self._fitted = True

        preds = self._model.predict(X_scaled).astype(int)
        accuracy = _compute_accuracy(y, preds)
        precision, recall, f1 = _compute_precision_recall_f1(y, preds)
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
        scaled = self._scale_vector(vector)
        return self._predict_from_scaled(scaled)

    def predict(self, features: Sequence[float]) -> int:
        _, p_llm = self.predict_proba(features)
        return int(p_llm >= 0.5)

    def save(self, path: Path | str) -> None:
        if not self._fitted or self.feature_names is None:
            raise RuntimeError("cannot save an unfitted model")
        if self._coefficients is None or self._intercept is None or self._classes is None:
            raise RuntimeError("model is missing coefficient data")
        if self._scaler_mean is None or self._scaler_scale is None:
            raise RuntimeError("model is missing scaler statistics")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": SERIALIZATION_FORMAT,
            "version": SERIALIZATION_VERSION,
            "feature_names": self.feature_names,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "scaler": {
                "mean": [float(x) for x in self._scaler_mean],
                "scale": [float(x) for x in self._scaler_scale],
            },
            "model": {
                "coefficients": [float(x) for x in self._coefficients],
                "intercept": float(self._intercept),
                "classes": [int(x) for x in self._classes],
            },
        }
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: Path | str) -> LogisticRegressionModel:
        path = Path(path)
        try:
            payload = cls._read_serialized_payload(path)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return cls._load_legacy_joblib(path)
        return cls._from_serialized_payload(payload)

    @staticmethod
    def _read_serialized_payload(path: Path) -> dict[str, Any]:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("serialized logistic model must be a JSON object")
        return payload

    @classmethod
    def _from_serialized_payload(cls, payload: dict[str, Any]) -> LogisticRegressionModel:
        if payload.get("format") != SERIALIZATION_FORMAT:
            raise ValueError("unsupported logistic model format")
        version = int(payload.get("version", 0))
        if version != SERIALIZATION_VERSION:
            raise ValueError(f"unsupported logistic model version: {version}")

        scaler_info = payload.get("scaler")
        model_info = payload.get("model")
        if not isinstance(scaler_info, dict) or not isinstance(model_info, dict):
            raise ValueError("serialized model payload is missing scaler/model sections")

        obj = cls.__new__(cls)
        obj.class_weight = payload.get("class_weight", "balanced")
        obj.max_iter = int(payload.get("max_iter", 1000))
        obj.random_state = int(payload.get("random_state", 42))
        feature_names = payload.get("feature_names")
        obj.feature_names = list(feature_names) if feature_names is not None else None

        obj._scaler = None
        obj._model = None
        obj._fitted = True

        try:
            mean_values = scaler_info["mean"]
            scale_values = scaler_info["scale"]
            coef_values = model_info["coefficients"]
            intercept_value = model_info["intercept"]
            classes_value = model_info["classes"]
        except KeyError as err:
            raise ValueError("serialized model payload is missing required fields") from err

        obj._scaler_mean = [float(x) for x in mean_values]
        obj._scaler_scale = [float(x) for x in scale_values]
        obj._coefficients = [float(x) for x in coef_values]
        obj._intercept = float(intercept_value)
        obj._classes = [int(x) for x in classes_value]

        expected_length = len(obj.feature_names) if obj.feature_names is not None else None
        if expected_length is not None and len(obj._coefficients) != expected_length:
            raise ValueError("coefficient vector length does not match feature names")
        if len(obj._coefficients) == 0:
            raise ValueError("serialized model is missing coefficients")
        if len(obj._scaler_mean) != len(obj._coefficients) or len(obj._scaler_scale) != len(obj._coefficients):
            raise ValueError("serialized scaler statistics are inconsistent")
        if len(obj._classes) != 2:
            raise ValueError("serialized model must provide two class labels")

        return obj

    @classmethod
    def _load_legacy_joblib(cls, path: Path) -> LogisticRegressionModel:
        try:
            import joblib  # type: ignore
        except ImportError as exc:  # pragma: no cover - legacy compatibility only.
            msg = "joblib is required to load legacy logistic regression models"
            raise RuntimeError(msg) from exc

        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError("legacy logistic regression model has unexpected structure")

        obj = cls.__new__(cls)
        obj.class_weight = payload.get("class_weight", "balanced")
        obj.max_iter = payload.get("max_iter", 1000)
        obj.random_state = payload.get("random_state", 42)
        feature_names = payload.get("feature_names")
        obj.feature_names = list(feature_names) if feature_names is not None else None

        obj._scaler = payload.get("scaler")
        obj._model = payload.get("model")
        obj._fitted = True

        if obj._scaler is not None and hasattr(obj._scaler, "mean_"):
            obj._scaler_mean = obj._scaler.mean_.astype(float).tolist()
            obj._scaler_scale = obj._scaler.scale_.astype(float).tolist()
        else:
            obj._scaler_mean = None
            obj._scaler_scale = None

        if obj._model is not None and hasattr(obj._model, "coef_"):
            obj._coefficients = obj._model.coef_.astype(float).ravel().tolist()
            obj._intercept = float(obj._model.intercept_.ravel()[0])
            obj._classes = [int(cls_val) for cls_val in obj._model.classes_]
        else:
            obj._coefficients = None
            obj._intercept = None
            obj._classes = None

        if (
            obj._coefficients is None
            or obj._intercept is None
            or obj._classes is None
            or obj._scaler_mean is None
            or obj._scaler_scale is None
        ):
            raise ValueError("legacy model payload is missing required fitted parameters")

        return obj

    def evaluate(self, dataset: FeatureDataset) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("model must be trained before evaluation")
        if not dataset.matrix:
            raise ValueError("evaluation dataset is empty")

        if self._model is not None:
            X = self._as_array(dataset.matrix)
            np_mod = _ensure_numpy()
            y = np_mod.asarray(dataset.labels, dtype=int)
            scaled = self._scaler.transform(X)
            probs = self._model.predict_proba(scaled)
            p_llm = probs[:, 1]
            preds = (p_llm >= 0.5).astype(int)
        else:
            y = [int(label) for label in dataset.labels]
            p_llm: list[float] = []
            preds: list[int] = []
            for row in dataset.matrix:
                _, p_llm_val = self.predict_proba(row)
                p_llm.append(p_llm_val)
                preds.append(int(p_llm_val >= 0.5))

        accuracy = _compute_accuracy(y, preds)
        precision, recall, f1 = _compute_precision_recall_f1(y, preds)
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


__all__ = ["LogisticRegressionModel", "TrainingResult"]
