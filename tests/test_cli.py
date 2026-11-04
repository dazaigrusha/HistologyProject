"""Перевірки окремого запуску методів і числового експорту LBP."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from histology.cli import run_method
from histology.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    """Сценарії не потребують дисплея чи Tkinter-вікна."""

    def setUp(self):
        self.config = load_config(ROOT / 'config.json')
        self.config['segmentation'].update(opening_radius=0, min_area=1)

    def test_all_methods_and_lbp_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((9, 9, 3), 255, dtype=np.uint8)
            image[2:7, 2:7] = [100, 45, 130]
            source = root / 'source.png'; Image.fromarray(image).save(source)
            mask = root / 'mask.png'
            binary = np.zeros((9, 9), dtype=np.uint8); binary[2:7, 2:7] = 255
            Image.fromarray(binary).save(mask)
            for method in ('stains', 'otsu', 'opening', 'watershed', 'lbp', 'all'):
                folder = run_method(mask if method in ('opening', 'watershed') else source,
                                    method, root / 'output', self.config)
                metadata = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['method'], method)
                if method in ('lbp', 'all'):
                    with (folder / 'lbp_histogram.csv').open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 256)
                    self.assertEqual(sum(int(item['count']) for item in rows), 49)
                    features = json.loads((folder / 'lbp_features.json').read_text(encoding='utf-8'))
                    self.assertEqual(features['features']['valid_centers'], 49)
                    self.assertTrue((folder / 'lbp_histogram.png').is_file())
                if method == 'all':
                    self.assertTrue((folder / 'objects.csv').is_file())
                    self.assertEqual(features['segmentation_stage'], 'watershed')

    def test_colored_input_rejected_for_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'color.png'
            Image.new('RGB', (5, 5), 'red').save(source)
            with self.assertRaises(ValueError):
                run_method(source, 'opening', root / 'output', self.config)

    def test_cli_subprocess_without_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'test.png'; Image.new('RGB', (3, 3), 'white').save(source)
            command = [sys.executable, str(ROOT / 'main.py'), '--input', str(source),
                       '--method', 'lbp', '--output', str(Path(tmp) / 'out')]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(list((Path(tmp) / 'out').glob('*/lbp_features.json')))
            invalid = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--method', 'lbp'],
                                     capture_output=True, text=True, timeout=20)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
