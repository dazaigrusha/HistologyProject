"""Пакетний запуск з еталоном, без еталона та з несправними файлами."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from histology.batch import run_batch
from histology.config import load_config
from histology.evaluation import evaluate_run
from histology.cli import run_method

ROOT = Path(__file__).resolve().parents[1]


class BatchTests(unittest.TestCase):
    """Помилка одного кадру не має знищувати зведений звіт."""

    def test_manifest_mixed_statuses(self):
        config = load_config(ROOT / 'config.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (5, 5), 'white').save(root / 'empty.png')
            np.save(root / 'truth.npy', np.zeros((5, 5), dtype=np.int32))
            np.save(root / 'wrong.npy', np.zeros((4, 4), dtype=np.int32))
            manifest = root / 'dataset.csv'
            manifest.write_text('image,reference\nempty.png,truth.npy\nempty.png,\nmissing.png,\nempty.png,wrong.npy\n', encoding='utf-8')
            folder = run_batch(manifest, root / 'output', config)
            summary = json.loads((folder / 'batch_summary.json').read_text())
            self.assertEqual(summary['rows'], 4)
            self.assertEqual(summary['processed'], 3)
            self.assertEqual(summary['evaluated'], 1)
            self.assertEqual(summary['processing_errors'], 1)
            self.assertEqual(summary['evaluation_errors'], 1)
            with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 4)  # Оцу, відкриття, до WS, фінал.
            self.assertTrue(all(float(row['dice']) == 1 for row in metrics))
            self.assertTrue((folder / 'paired_comparison.csv').exists())
            self.assertTrue(all(group['mean_dice_nonempty_reference'] is None for group in summary['stages']))

    def test_binary_reference_has_no_count_error(self):
        config = load_config(ROOT / 'config.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'empty.png'; Image.new('RGB', (5, 5), 'white').save(path)
            reference = root / 'truth.png'; Image.new('L', (5, 5), 0).save(reference)
            folder = run_method(path, 'all', root / 'output', config)
            evaluation = evaluate_run(folder, reference, config['max_pixels'])
            self.assertTrue(all(row['reference_count'] is None for row in evaluation['rows']))
            self.assertTrue(all(row['count_absolute_error'] is None for row in evaluation['rows']))

    def test_empty_manifest_rejected(self):
        config = load_config(ROOT / 'config.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.csv'; path.write_text('image,reference\n')
            with self.assertRaises(ValueError):
                run_batch(path, Path(tmp) / 'out', config)


if __name__ == '__main__':
    unittest.main()
