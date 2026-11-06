# Method and limits

## Question and data

Could an offline chainsaw sound classifier reduce false alarms relative to a simple spectral rule? The data are four ESC-50 classes: chainsaw (positive) and hand saw, engine, and helicopter (negative), each with 40 five-second clips. The source is pinned to commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. The downloader obtains original WAVs and metadata from the [ESC-50 repository](https://github.com/karolpiczak/ESC-50). The dataset is CC BY-NC; source attribution is retained in its metadata and upstream LICENSE. No clips are committed here.

## Signal processing

WAV is downmixed if needed, converted to floating point, resampled to 16 kHz, mean-centered, and normalized by RMS. For each 512-sample STFT frame with 256-sample overlap, the feature extractor computes centroid, spectral flatness, and eight normalized frequency-band energies. It also computes 12 MFCCs from 26 triangular mel filters. Mean and standard deviation across frames give 44 clip-level features. Level normalization deliberately removes absolute loudness, which may be useful or harmful depending on the deployment microphone.

## Evaluation protocol

The dataset's five predefined folds keep slices of one original recording together. For each outer fold, 128 clips are training data and 32 are held out. The rule's centroid cutoff is optimized on training clips for balanced accuracy. The logistic classifier uses `StandardScaler` and class-balanced `LogisticRegression`. The false-alarm gate selects a probability-score threshold from four **inner training folds** so that the inner out-of-fold negative false-positive rate is at most 10%, then fits the classifier on all 128 outer-training clips and evaluates once on the 32 outer-test clips. Neither feature scaling nor threshold selection uses the outer-test fold. The class list and model family were chosen for this project; the held-out fold is not an untouched external validation set.

Pooled results use `TP` = chainsaw flagged, `FN` = chainsaw missed, `FP` = a negative clip flagged, and `TN` = a negative clip ignored. Precision = TP/(TP+FP), recall = TP/(TP+FN), and false-positive rate = FP/(FP+TN). Counts are **per five-second clip**. There is no temporal alert-persistence step or false-alarms-per-hour measurement because ESC-50 does not provide continuous monitoring recordings. The one-clip command retrains on all selected clips for demonstration; it is not an independent evaluation.

This is an **exploratory** benchmark. The feature set was revised after inspecting an initial cross-validation result. Consequently, the final five-fold scores are useful for comparing these three methods within this project, but they are not a locked, unbiased estimate of future performance. An external dataset is required for that.

## Threats to validity

- Four small balanced classes do not approximate the prevalence, acoustics, microphone placement, or noise of a forest monitoring site.
- ESC-50's authors [warn about class-dependent preprocessing that could leak label information](https://github.com/karolpiczak/ESC-50#caveats). The folds protect recording identity but cannot remove that issue.
- The gated model's 10% training false-positive ceiling is not a guarantee on held-out clips; calibration can shift when refitted on all training clips.
- Score thresholds and metrics depend on these four selected classes. Other machinery, wind, overlapping sounds, and unseen recording devices are not tested.
- This implementation is offline and uses no microphone, live network, edge device, generative AI, or real-time latency measurement.

## Next research step

Collect or obtain a continuous, appropriately licensed recording set from different locations and devices. Freeze the model and threshold before testing it. Report event-level detection delay, false alarms per recording hour, and recall with uncertainty intervals. A larger dataset and a less aggressive threshold would be needed before considering an operator alert demo.
