import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .utils import label_to_class, class_name


def train_random_forest(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None,
                        n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y, sample_weight=sample_weights)
    return model


def predict(model: RandomForestClassifier, X: np.ndarray) -> list[dict]:
    probs = model.predict_proba(X)
    preds = model.predict(X)
    results = []
    for i in range(len(preds)):
        label = int(preds[i])
        code = label_to_class(label)
        confidence = float(probs[i][label])
        results.append({
            "class_code": code,
            "class_name": class_name(code),
            "confidence": round(confidence, 4),
        })
    return results


def save_rf(model: RandomForestClassifier, path: str) -> None:
    joblib.dump(model, path)


def load_rf(path: str) -> RandomForestClassifier:
    return joblib.load(path)
