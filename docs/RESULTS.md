# Measured ESC-50 result snapshot

Run: 6 November 2026, Python 3.14.5, NumPy 2.4.5, SciPy 1.17.1, scikit-learn 1.8.0. The source uses 160 ESC-50 clips, 40 per selected class, and five official folds. All reported counts are on held-out clips, pooled across folds.

| Method | TP | FP | TN | FN | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centroid rule | 34 | 60 | 60 | 6 | 0.362 | 0.850 | 0.500 |
| Logistic at 0.5 | 23 | 27 | 93 | 17 | 0.460 | 0.575 | 0.225 |
| Logistic with train-selected false-alarm gate | 11 | 11 | 109 | 29 | 0.500 | 0.275 | 0.092 |

The strict gate flags 11 of 40 chainsaws and incorrectly flags 11 of 120 negative clips. Of those 11 false positives, six are hand saw, three helicopter, and two engine. It reduces false alarms but sacrifices too much recall for a reliable monitoring system. This negative result is useful: selecting a very low target FPR on this small dataset does not produce a practical alarm threshold.

The figures do **not** measure false alerts per hour, safety performance, detection of illegal logging, or performance on live microphones. The model's numeric output is an uncalibrated score. See [method and limits](METHOD.md).
