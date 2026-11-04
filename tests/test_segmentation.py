"""Ручні еталони порога, морфології та сегментації."""
import unittest
import numpy as np
from histology.segmentation import binary_opening, otsu_threshold, validate_segmentation
from histology.analysis import segment_nuclei


class SegmentationTests(unittest.TestCase):
    """Короткі перевірки без готових CV-алгоритмів."""

    def test_otsu_two_levels(self):
        image = np.array([[0, 0, 0], [0, 100, 100], [100, 100, 100]], dtype=np.uint8)
        # Усі t=0..99 однаково розділяють два рівні; правило — найменший t.
        self.assertEqual(otsu_threshold(image), 0)
        np.testing.assert_array_equal(image > otsu_threshold(image), image == 100)

    def test_otsu_nontrivial_split(self):
        # Для 0,1,4: scores(t=0)=12.5; scores(t=1,2,3)=24.5.
        image = np.array([[0, 1, 4]], dtype=np.uint8)
        self.assertEqual(otsu_threshold(image), 1)

    def test_constant_image_has_no_foreground(self):
        for level in (0, 100, 255):
            image = np.full((3, 3), level, dtype=np.uint8)
            self.assertFalse(np.any(image > otsu_threshold(image)))

    def test_opening_square_becomes_cross(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        expected = np.zeros_like(mask)
        expected[2, 1:4] = True
        expected[1:4, 2] = True
        np.testing.assert_array_equal(binary_opening(mask, 1), expected)

    def test_opening_removes_speck_and_zero_is_identity(self):
        mask = np.eye(5, dtype=bool)
        self.assertFalse(np.any(binary_opening(mask, 1)))
        np.testing.assert_array_equal(binary_opening(mask, 0), mask)

    def test_opening_border_is_background(self):
        self.assertFalse(np.any(binary_opening(np.ones((1, 1), dtype=bool), 1)))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            otsu_threshold(np.zeros((2, 2), dtype=float))
        with self.assertRaises(ValueError):
            binary_opening(np.zeros((2, 2), dtype=np.uint8), 1)
        with self.assertRaises(ValueError):
            validate_segmentation({})

    def test_full_synthetic_segmentation(self):
        settings = {'threshold_floor': 15, 'opening_radius': 1, 'min_area': 5,
                    'max_area': 0, 'connectivity': 8, 'exclude_border': False}
        h_map = np.zeros((9, 13), dtype=np.float32)
        h_map[2:5, 2:5] = -np.log(55 / 255)
        h_map[2:5, 8:11] = -np.log(55 / 255)
        h_map[7, 6] = -np.log(55 / 255)
        result = segment_nuclei(h_map, settings)
        self.assertEqual(result['count'], 2)
        self.assertEqual(int(result['raw_mask'].sum()), 19)
        self.assertEqual(int(result['mask'].sum()), 10)
        self.assertEqual([item['area_px'] for item in result['objects']], [5, 5])
        self.assertEqual(result['effective_threshold'], 15)
        np.testing.assert_array_equal(result['labels'] > 0, result['mask'])

    def test_no_h_returns_empty(self):
        settings = {'threshold_floor': 0, 'opening_radius': 0, 'min_area': 1,
                    'max_area': 0, 'connectivity': 8, 'exclude_border': False}
        result = segment_nuclei(np.zeros((5, 5), dtype=np.float32), settings)
        self.assertEqual(result['count'], 0)
        self.assertFalse(np.any(result['mask']))


if __name__ == '__main__':
    unittest.main()
