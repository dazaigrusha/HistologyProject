"""Ручні еталони площі, сіткового периметра та одиниць вимірювання."""
import math
import unittest
import numpy as np
from histology.measurements import filter_regions, measure_regions, summarize_regions
from histology.analysis import add_measurements, apply_watershed, segment_nuclei


class MeasurementTests(unittest.TestCase):
    """Вимірювання мають відповідати збереженим міткам."""

    def setUp(self):
        self.seg = {'threshold_floor': 15, 'opening_radius': 0, 'min_area': 1,
                    'max_area': 0, 'connectivity': 8, 'exclude_border': False}

    def test_square_and_calibration(self):
        labels = np.pad(np.ones((2, 2), dtype=np.int32), 1)
        item = measure_regions(labels, .5)[0]
        self.assertEqual(item['area_px'], 4)
        self.assertEqual(item['perimeter_px'], 8)
        self.assertAlmostEqual(item['circularity'], math.pi / 4)
        self.assertEqual(item['area_um2'], 1)
        self.assertEqual(item['perimeter_um'], 4)
        self.assertEqual(item['centroid_x'], 1.5)
        summary = summarize_regions([item], labels.shape, .5)
        self.assertEqual(summary['density_per_megapixel'], 62500)
        self.assertEqual(summary['density_per_mm2'], 250000)

    def test_hole_and_shared_boundary(self):
        ring = np.ones((3, 3), dtype=np.int32); ring[1, 1] = 0
        item = measure_regions(ring)[0]
        self.assertEqual(item['area_px'], 8)
        self.assertEqual(item['perimeter_px'], 16)  # Зовнішні 12 + внутрішні 4.
        objects = measure_regions(np.array([[1, 2]], dtype=np.int32))
        self.assertEqual([item['perimeter_px'] for item in objects], [4, 4])

    def test_filter_keeps_adjacent_ids_separate(self):
        labels = np.array([[1, 2, 2]], dtype=np.int32)
        self.seg['min_area'] = 2
        filtered, objects = filter_regions(labels, self.seg)
        np.testing.assert_array_equal(filtered, [[0, 1, 1]])
        self.assertEqual(objects[0]['area_px'], 2)
        self.assertEqual(objects[0]['id'], 1)

    def test_empty_summary(self):
        objects = measure_regions(np.zeros((3, 3), dtype=np.int32))
        summary = summarize_regions(objects, (3, 3))
        self.assertEqual(summary['density_per_megapixel'], 0)
        self.assertIsNone(summary['mean_circularity'])
        self.assertIsNone(summary['density_per_mm2'])

    def test_analysis_integration_and_repeat(self):
        yy, xx = np.indices((29, 39))
        mask = ((yy - 14) ** 2 + (xx - 13) ** 2 <= 64) | ((yy - 14) ** 2 + (xx - 25) ** 2 <= 64)
        h_map = mask.astype(np.float32) * 2
        before = add_measurements(segment_nuclei(h_map, self.seg), {'pixel_size_um': None})
        self.assertEqual(before['count'], 1)
        ws = {'min_peak_distance': 8, 'min_peak_height': 2, 'relative_peak_height': .35}
        after = apply_watershed(before, self.seg, ws, {'pixel_size_um': None})
        self.assertEqual(after['before_count'], 1)
        self.assertEqual(after['count'], 2)
        self.assertEqual(sum(item['area_px'] for item in after['objects']), int(mask.sum()))
        again = apply_watershed(after, self.seg, ws, {'pixel_size_um': None})
        self.assertEqual(again['before_count'], 1)
        np.testing.assert_array_equal(after['labels'], again['labels'])
        for item in after['objects']:
            self.assertEqual(item['area_px'], int(np.sum(after['labels'] == item['id'])))

    def test_large_merged_component_is_split_before_max_area_filter(self):
        yy, xx = np.indices((29, 39))
        mask = ((yy - 14) ** 2 + (xx - 13) ** 2 <= 64) | ((yy - 14) ** 2 + (xx - 25) ** 2 <= 64)
        self.seg['max_area'] = 250
        before = segment_nuclei(mask.astype(np.float32) * 2, self.seg)
        self.assertEqual(before['count'], 0)
        after = apply_watershed(before, self.seg,
            {'min_peak_distance': 8, 'min_peak_height': 2, 'relative_peak_height': .35},
            {'pixel_size_um': None})
        self.assertEqual(after['count'], 2)
        self.assertTrue(all(item['area_px'] <= 250 for item in after['objects']))


if __name__ == '__main__':
    unittest.main()
