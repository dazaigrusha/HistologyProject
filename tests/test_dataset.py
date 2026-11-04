"""Перевірки імпорту instance-TIFF, зіставлення пар та розділення груп."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from histology.dataset import prepare_dataset, read_instance_mask


class DatasetTests(unittest.TestCase):
    """ID понад 255 не можна втрачати при конвертації розмітки."""

    def test_uint16_ids_and_group_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / 'raw'; masks = root / 'masks'
            images.mkdir(); masks.mkdir()
            for name in ('p01_a', 'p01_b', 'p02_a', 'p03_a'):
                Image.new('RGB', (5, 5), 'white').save(images / f'{name}.png')
                labels = np.zeros((5, 5), dtype=np.uint16)
                labels[1, 1] = 300; labels[3, 3] = 1000
                Image.fromarray(labels).save(masks / f'{name}.tif')
            folder = prepare_dataset(images, masks, root / 'out', group_regex=r'^(p\d+)_')
            data = json.loads((folder / 'dataset.json').read_text())
            self.assertEqual(data['images'], 4)
            p01 = [row['split'] for row in data['items'] if row['group'] == 'p01']
            self.assertEqual(len(set(p01)), 1)
            self.assertTrue(data['calibration_images'] > 0 and data['test_images'] > 0)
            loaded = np.load(folder / 'references/p01_a.npy', allow_pickle=False)
            self.assertEqual(int(loaded[3, 3]), 1000)
            for split in ('calibration', 'test', 'all'):
                with (folder / f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
                    for row in csv.DictReader(stream):
                        self.assertTrue((folder / row['image']).exists())
                        self.assertTrue((folder / row['reference']).exists())
            again = prepare_dataset(images, masks, root / 'out', group_regex=r'^(p\d+)_')
            second = json.loads((again / 'dataset.json').read_text())
            self.assertEqual([(row['name'], row['split']) for row in data['items']],
                             [(row['name'], row['split']) for row in second['items']])

    def test_missing_pair_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'images').mkdir(); (root / 'masks').mkdir()
            Image.new('RGB', (5, 5)).save(root / 'images/a.png')
            with self.assertRaises(ValueError):
                prepare_dataset(root / 'images', root / 'masks', root / 'out')

    def test_color_mask_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mask.png'; Image.new('RGB', (5, 5)).save(path)
            with self.assertRaises(ValueError):
                read_instance_mask(path, 100)


if __name__ == '__main__':
    unittest.main()
