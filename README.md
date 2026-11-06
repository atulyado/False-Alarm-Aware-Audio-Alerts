# False-Alarm-Aware Audio Alerts

**Research question:** Can a lightweight audio classifier reduce false chainsaw alerts compared with a single spectral rule, and what recall is lost when its alert threshold is tightened?

This independent portfolio project is inspired by Prof. Shigeru Kashihara's [work on sound-monitoring false alarms and audio alerts](https://research-db.oit.ac.jp/html/200000172_ronbn_1_en.html). It is not affiliated with, assigned by, or evaluated by his lab. It is an **offline clip-classification experiment**, not a live forest monitor or safety system.

## Reproduce

Use Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
python -m audio_alerts.dataset
python -m audio_alerts.benchmark
python -m unittest discover -s tests -v
```

After activating the virtual environment, the `python` commands above work on either platform. The dataset command downloads **only 160 clips** into ignored `data/esc50/`: 40 each of chainsaw, hand saw, engine, and helicopter, pinned to [ESC-50 commit `33c8ce9`](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6). No audio is redistributed in this repository. The original dataset is [CC BY-NC](https://github.com/karolpiczak/ESC-50#license); credit and use restrictions remain with its authors and source recordings.

The benchmark writes `results/predictions.csv` and `results/summary.json`. The CSV lists each held-out clip, class, predicted label, and model score. These files are ignored by Git; a [result snapshot](docs/RESULTS.md) is committed for inspection. To score one WAV file as a **research demo** after downloading the data:

```bash
python -m audio_alerts.predict path/to/clip.wav
```

## Method

The positive class is **chainsaw**; the other three classes are negatives. WAV files are converted to mono, resampled to 16 kHz, centered and level-normalized. Each clip yields 44 features: means and standard deviations of spectral centroid, flatness, eight normalized spectral bands, and 12 MFCCs. No pretrained model or cloud API is used.

Three decisions are compared on the same [official five ESC-50 folds](https://github.com/karolpiczak/ESC-50#overview):

1. **Centroid rule:** A single spectral-centroid threshold and direction selected using the outer training folds only.
2. **Logistic classifier:** Standardized features, class-balanced logistic regression, and a 0.5 score threshold.
3. **False-alarm gate:** The same classifier with a threshold selected from **inner out-of-fold scores on the training portion only**. It maximizes training recall while targeting at most 10% training false-positive rate. The outer test fold is untouched until evaluation.

Each outer fold trains on 128 clips and tests on 32. The five held-out predictions are pooled into one confusion matrix per method. The gate's 10% setting is a **training target, not a guarantee on held-out or field audio**. Details and limitations are in [methods](docs/METHOD.md).

This is an **exploratory** comparison: features were improved after examining an initial cross-validation run. These scores should not be presented as an unbiased estimate for unseen recordings.

## Measured five-fold results

| Method | TP | FP | TN | FN | Precision | Recall | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centroid rule | 34 | 60 | 60 | 6 | 0.362 | 0.850 | 0.500 |
| Logistic, threshold 0.5 | 23 | 27 | 93 | 17 | 0.460 | 0.575 | 0.225 |
| False-alarm gate | 11 | 11 | 109 | 29 | 0.500 | 0.275 | 0.092 |

**Interpretation:** The gate reduces false positives from 60 to 11 versus the simple rule, but misses 29 of 40 chainsaw clips. This setting is therefore **too insensitive for any safety-critical use**. The project demonstrates the tradeoff rather than claiming a successful detector. See the [per-class error analysis and limitations](docs/RESULTS.md).

## Repository map

```text
audio_alerts/dataset.py    Pinned ESC-50 subset downloader
audio_alerts/features.py   WAV processing and spectral/MFCC features
audio_alerts/benchmark.py  Nested threshold selection and held-out evaluation
audio_alerts/predict.py    One-clip research demo
tests/                     Data and evaluation tests
docs/METHOD.md             Experimental design and limitations
docs/RESULTS.md            Measured result snapshot and error analysis
```

**Resume description:** Built a reproducible chainsaw-alert benchmark on 160 ESC-50 clips using five held-out folds; compared a spectral rule with a logistic classifier and a training-selected false-alarm threshold, documenting the reduction in false positives alongside the loss in recall.
