"""Five-fold ESC-50 chainsaw alert benchmark with a train-only false-alarm gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import CLASSES
from .features import extract


def load_data(root: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    metadata = root / "selected.csv"
    if not metadata.exists():
        raise FileNotFoundError("Run python -m audio_alerts.dataset first")
    with metadata.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 160 or {r["category"] for r in rows} != set(CLASSES):
        raise ValueError("Expected exactly 40 clips in each of four selected ESC-50 classes")
    features = []
    for i, row in enumerate(rows, 1):
        path = root / "audio" / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        features.append(extract(path))
        if i % 40 == 0:
            print(f"Extracted features {i}/{len(rows)}")
    x = np.asarray(features)
    y = np.asarray([int(r["category"] == "chainsaw") for r in rows])
    folds = np.asarray([int(r["fold"]) for r in rows])
    if not np.isfinite(x).all() or set(folds) != set(range(1, 6)):
        raise ValueError("Invalid features or fold labels")
    return rows, x, y, folds


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def rule_fit(scores: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Tune one spectral-centroid cutoff on training clips only."""
    thresholds = np.r_[scores.min() - 1e-9, np.unique(scores), scores.max() + 1e-9]
    best = (-1.0, 1, 0.0)
    for direction in (1, -1):
        for threshold in thresholds:
            prediction = (direction * scores >= direction * threshold).astype(int)
            tpr = prediction[y == 1].mean()
            tnr = 1 - prediction[y == 0].mean()
            candidate = ((tpr + tnr) / 2, direction, float(threshold))
            if candidate[0] > best[0]:
                best = candidate
    return best[1], best[2]


def alert_gate_threshold(y: np.ndarray, probability: np.ndarray, max_fpr: float) -> float:
    """Maximize training recall under a training false-positive-rate ceiling."""
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1")
    candidates = np.r_[0.0, np.unique(probability), 1.0, np.nextafter(1.0, 2.0)]
    viable = []
    for threshold in candidates:
        predicted = probability >= threshold
        fpr = predicted[y == 0].mean()
        if fpr <= max_fpr + 1e-12:
            recall = predicted[y == 1].mean()
            viable.append((recall, -fpr, -float(threshold), float(threshold)))
    return max(viable)[3]


def inner_oof(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold scores inside the outer training set; no outer-test tuning."""
    result = np.zeros(len(y))
    for fold in np.unique(folds):
        train, validation = folds != fold, folds == fold
        fitted = model().fit(x[train], y[train])
        result[validation] = fitted.predict_proba(x[validation])[:, 1]
    return result


def metrics(rows: list[dict], method: str) -> dict:
    selected = [r for r in rows if r["method"] == method]
    y = [r["actual"] for r in selected]
    predicted = [r["predicted"] for r in selected]
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"method": method, "clips": len(selected), "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "precision": round(precision_score(y, predicted, zero_division=0), 3),
            "recall": round(recall_score(y, predicted, zero_division=0), 3),
            "false_positive_rate": round(fp / (fp + tn), 3)}


def evaluate(rows: list[dict[str, str]], x: np.ndarray, y: np.ndarray,
             folds: np.ndarray, max_fpr: float = 0.10) -> tuple[list[dict], list[dict]]:
    if len(x) != len(y) or len(x) != len(folds) or len(rows) != len(y):
        raise ValueError("Data lengths must match")
    predictions = []
    for fold in range(1, 6):
        train, test = folds != fold, folds == fold
        direction, cutoff = rule_fit(x[train, 0], y[train])
        oof = inner_oof(x[train], y[train], folds[train])
        threshold = alert_gate_threshold(y[train], oof, max_fpr)
        fitted = model().fit(x[train], y[train])
        probability = fitted.predict_proba(x[test])[:, 1]
        for index, p in zip(np.where(test)[0], probability):
            for method, predicted in (
                ("centroid_rule", int(direction * x[index, 0] >= direction * cutoff)),
                ("logistic_0_5", int(p >= 0.5)),
                ("false_alarm_gate", int(p >= threshold)),
            ):
                predictions.append({"filename": rows[index]["filename"], "class": rows[index]["category"],
                                    "fold": fold, "actual": int(y[index]), "method": method,
                                    "predicted": predicted, "probability": round(float(p), 6),
                                    "decision_threshold": round(float(threshold), 6) if method == "false_alarm_gate" else ""})
        print(f"Held-out fold {fold}: rule cutoff={cutoff:.3f}, gated model cutoff={threshold:.3f}")
    reports = [metrics(predictions, method) for method in ("centroid_rule", "logistic_0_5", "false_alarm_gate")]
    return predictions, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/esc50"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-train-fpr", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.max_train_fpr <= 1:
        parser.error("--max-train-fpr must be between 0 and 1")
    started = perf_counter()
    rows, x, y, folds = load_data(args.data)
    predictions, reports = evaluate(rows, x, y, folds, args.max_train_fpr)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result = {"source": "ESC-50", "classes": list(CLASSES), "folds": 5,
              "max_training_fpr": args.max_train_fpr, "feature_count": int(x.shape[1]),
              "elapsed_seconds": round(perf_counter() - started, 2), "methods": reports}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
