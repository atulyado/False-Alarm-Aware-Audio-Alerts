"""Research demo: score one WAV clip with a model fitted on the selected ESC-50 subset."""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import alert_gate_threshold, inner_oof, load_data, model
from .features import extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="WAV clip of at least one second")
    parser.add_argument("--data", type=Path, default=Path("data/esc50"))
    parser.add_argument("--max-train-fpr", type=float, default=0.10)
    args = parser.parse_args()
    rows, x, y, folds = load_data(args.data)
    if not 0 <= args.max_train_fpr <= 1:
        parser.error("--max-train-fpr must be between 0 and 1")
    threshold = alert_gate_threshold(y, inner_oof(x, y, folds), args.max_train_fpr)
    probability = float(model().fit(x, y).predict_proba(extract(args.wav).reshape(1, -1))[0, 1])
    print(f"Chainsaw score: {probability:.3f}")
    print(f"Training-derived alert threshold: {threshold:.3f}")
    print("ALERT" if probability >= threshold else "NO ALERT")
    print("Research demo only; this score is not a calibrated real-world probability.")


if __name__ == "__main__":
    main()
