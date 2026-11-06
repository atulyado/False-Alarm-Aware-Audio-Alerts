import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

from audio_alerts.benchmark import alert_gate_threshold, metrics, rule_fit
from audio_alerts.dataset import selected_rows
from audio_alerts.features import extract


class BenchmarkTests(unittest.TestCase):
    def test_gate_uses_negative_ceiling(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.4, 0.8, 0.1])
        threshold = alert_gate_threshold(y, scores, 0.0)
        self.assertEqual((scores[y == 0] >= threshold).sum(), 0)
        self.assertEqual((scores[y == 1] >= threshold).sum(), 1)

    def test_centroid_rule_finds_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        direction, cutoff = rule_fit(scores, y)
        self.assertTrue(np.array_equal((direction * scores >= direction * cutoff).astype(int), y))

    def test_metrics_count_false_alarms(self):
        rows = [{"method": "rule", "actual": actual, "predicted": predicted}
                for actual, predicted in ((1, 1), (1, 0), (0, 1), (0, 0))]
        result = metrics(rows, "rule")
        self.assertEqual((result["TP"], result["FP"], result["TN"], result["FN"]), (1, 1, 1, 1))
        self.assertEqual(result["false_positive_rate"], 0.5)

    def test_metadata_rejects_wrong_subset(self):
        with self.assertRaises(ValueError):
            selected_rows("filename,category,fold\na.wav,chainsaw,1\n")

    def test_extract_accepts_stereo_and_returns_finite_features(self):
        seconds = np.arange(16000, dtype=np.float64) / 16000
        signal = (12000 * np.sin(2 * np.pi * 440 * seconds)).astype(np.int16)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            wavfile.write(path, 16000, np.column_stack([signal, signal]))
            features = extract(path)
        self.assertEqual(features.shape, (44,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
