"""Еталони EDT, маркерів і розділення злитих областей."""
import unittest
import numpy as np
from histology.components import connected_components
from histology.watershed import automatic_markers, distance_transform, split_touching, watershed_flood


class WatershedTests(unittest.TestCase):
    """Малі маски з відомими відстанями та межами."""

    def setUp(self):
        self.settings = {'min_peak_distance': 4.0, 'min_peak_height': 2.0,
                         'relative_peak_height': .35}

    def test_edt_full_square_manual(self):
        # Фон є також за краєм; центр квадрата 5×5 має відстань 3.
        expected = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1],
                             [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]])
        np.testing.assert_allclose(distance_transform(np.ones((5, 5), dtype=bool)), expected)

    def test_edt_diagonal_and_brute_force(self):
        # Прямий повний перебір — незалежний еталон на малих масках.
        for seed in range(4):
            mask = np.random.default_rng(seed).random((5, 7)) > .25
            padded = np.pad(mask, 1)
            background = np.column_stack(np.nonzero(~padded))
            expected = np.zeros(mask.shape)
            for y, x in zip(*np.nonzero(mask)):
                expected[y, x] = np.sqrt(np.min(np.sum((background - [y + 1, x + 1]) ** 2, axis=1)))
            np.testing.assert_allclose(distance_transform(mask), expected, atol=1e-6)

    def test_empty_and_single_pixel(self):
        result = split_touching(np.zeros((3, 3), dtype=bool), self.settings)
        self.assertFalse(result['watershed_labels'].any())
        result = split_touching(np.ones((1, 1), dtype=bool), self.settings)
        self.assertEqual(result['distance'][0, 0], 1)
        self.assertEqual(result['markers'][0, 0], 1)  # Резервний маркер нижче min_peak_height.

    def test_flat_plateau_two_markers_manual(self):
        distance = np.ones((1, 5), dtype=np.float32)
        markers = np.array([[1, 0, 0, 0, 2]], dtype=np.int32)
        labels = watershed_flood(distance, markers, distance > 0, 4)
        np.testing.assert_array_equal(labels, [[1, 1, 1, 2, 2]])

    def test_flat_top_gets_one_marker(self):
        mask = np.ones((4, 4), dtype=bool)
        result = split_touching(mask, self.settings)
        self.assertEqual(int(result['markers'].max()), 1)
        self.assertEqual(int(result['watershed_labels'].max()), 1)

    def test_spacing_and_separate_components(self):
        distance = np.array([[1, 3, 1, 3, 1]], dtype=np.float32)
        components = np.ones((1, 5), dtype=np.int32)
        markers = automatic_markers(distance, components, self.settings)
        self.assertEqual(np.count_nonzero(markers), 1)
        mask = np.eye(3, dtype=bool)
        result = split_touching(mask, self.settings, connectivity=4)
        self.assertEqual(np.count_nonzero(result['markers']), 3)
        self.assertEqual(len(np.unique(result['watershed_labels'])) - 1, 3)

    def test_touching_disks_two_basins(self):
        yy, xx = np.indices((29, 39))
        mask = ((yy - 14) ** 2 + (xx - 13) ** 2 <= 64) | ((yy - 14) ** 2 + (xx - 25) ** 2 <= 64)
        _, objects = connected_components(mask)
        self.assertEqual(len(objects), 1)
        result = split_touching(mask, self.settings)
        self.assertEqual(int(result['markers'].max()), 2)
        np.testing.assert_array_equal(result['watershed_labels'] > 0, mask)
        self.assertNotEqual(result['watershed_labels'][14, 13], result['watershed_labels'][14, 25])
        again = split_touching(mask, self.settings)
        np.testing.assert_array_equal(result['watershed_labels'], again['watershed_labels'])

    def test_missing_marker_is_error(self):
        mask = np.ones((3, 3), dtype=bool)
        with self.assertRaises(ValueError):
            watershed_flood(distance_transform(mask), np.zeros((3, 3), dtype=np.int32), mask)


if __name__ == '__main__':
    unittest.main()
