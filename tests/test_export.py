"""Перевірка повного експорту та порожньої таблиці."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from histology.analysis import object_overlay, segment_nuclei
from histology.export import save_results
from histology.config import load_config


class ExportTests(unittest.TestCase):
    """Мітки, CSV і параметри повинні відповідати одній масці."""

    def test_export_and_unique_folders(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config.json')
        maps = (np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32))
        rgb = np.full((3, 3, 3), 255, dtype=np.uint8)
        result = segment_nuclei(maps[0], config['segmentation'])
        result['overlay'] = object_overlay(rgb, result['labels'], result['objects'])
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_results(Path(tmp), 'test.png', (3, 3), maps, config, result)
            other = save_results(Path(tmp), 'test.png', (3, 3), maps, config)
            self.assertNotEqual(folder, other)
            np.testing.assert_array_equal(np.load(folder / 'labels.npy', allow_pickle=False), result['labels'])
            metadata = json.loads((folder / 'parameters.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['candidate_count'], 0)
            with (folder / 'objects.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertTrue((folder / 'overlay.png').is_file())
            self.assertFalse((other / 'labels.npy').exists())

    def test_export_nonempty_labels_match_csv(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config.json')
        config['segmentation'].update(opening_radius=0, min_area=1)
        h_map = np.zeros((5, 5), dtype=np.float32)
        h_map[1, 1] = 2
        h_map[3, 3] = 2
        result = segment_nuclei(h_map, config['segmentation'])
        rgb = np.full((5, 5, 3), 255, dtype=np.uint8)
        result['overlay'] = object_overlay(rgb, result['labels'], result['objects'])
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_results(Path(tmp), 'two.png', (5, 5), (h_map, h_map), config, result)
            with (folder / 'objects.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            labels = np.load(folder / 'labels.npy', allow_pickle=False)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(int(row['area_px']), int(np.sum(labels == int(row['id']))))

    def test_watershed_workflow_exports_before_and_after(self):
        from histology.app import analyze_image, split_image
        config = load_config(Path(__file__).resolve().parents[1] / 'config.json')
        config['segmentation'].update(opening_radius=0, min_area=1)
        yy, xx = np.indices((29, 39))
        mask = ((yy - 14) ** 2 + (xx - 13) ** 2 <= 64) | ((yy - 14) ** 2 + (xx - 25) ** 2 <= 64)
        h_map = mask.astype(np.float32) * 2
        rgb = np.full((29, 39, 3), 255, dtype=np.uint8)
        before = analyze_image(rgb, h_map, config['segmentation'], config['measurements'])
        after = split_image(rgb, before, config)
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_results(Path(tmp), 'touching.png', (39, 29), (h_map, h_map), config, after)
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['before']['candidate_count'], 1)
            self.assertEqual(summary['after']['candidate_count'], 2)
            self.assertEqual(summary['markers'], 2)
            with (folder / 'objects.csv').open(encoding='utf-8-sig', newline='') as stream:
                objects = list(csv.DictReader(stream))
            self.assertEqual(len(objects), 2)
            labels = np.load(folder / 'labels.npy', allow_pickle=False)
            for item in objects:
                self.assertEqual(int(item['area_px']), int(np.sum(labels == int(item['id']))))
                self.assertGreater(float(item['perimeter_px']), 0)
                self.assertEqual(item['area_um2'], '')
            for filename in ('before_labels.npy', 'markers.npy', 'distance.npy',
                             'watershed_labels.npy', 'objects_before.csv', 'before_overlay.png'):
                self.assertTrue((folder / filename).is_file())

    def test_evaluation_is_exported_with_current_result(self):
        from histology.metrics import evaluate_result
        config = load_config(Path(__file__).resolve().parents[1] / 'config.json')
        h_map = np.zeros((3, 3), dtype=np.float32)
        result = segment_nuclei(h_map, config['segmentation'])
        result['overlay'] = object_overlay(np.full((3, 3, 3), 255, dtype=np.uint8),
                                           result['labels'], result['objects'])
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / 'truth.npy'
            np.save(truth, np.zeros((3, 3), dtype=np.int32))
            result['evaluation'] = evaluate_result(result, truth, 100)
            folder = save_results(Path(tmp), 'empty.png', (3, 3), (h_map, h_map), config, result)
            evaluation = json.loads((folder / 'evaluation.json').read_text())
            self.assertEqual(evaluation['rows'][0]['dice'], 1)
            self.assertTrue((folder / 'evaluation.csv').is_file())


if __name__ == '__main__':
    unittest.main()
