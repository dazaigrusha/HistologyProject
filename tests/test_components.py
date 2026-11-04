"""Перевірки міток, діагоналей, злиття та відбору областей."""
import unittest
import numpy as np
from histology.components import connected_components


class ComponentTests(unittest.TestCase):
    """Очікувані компоненти для малих масок задані вручну."""

    def test_diagonals(self):
        mask = np.eye(3, dtype=bool)
        labels, objects = connected_components(mask, 4)
        self.assertEqual(len(objects), 3)
        np.testing.assert_array_equal(labels, np.diag([1, 2, 3]))
        labels, objects = connected_components(mask, 8)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['area_px'], 3)
        self.assertEqual(objects[0]['centroid_x'], 1)

    def test_bridge_merges_labels(self):
        mask = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
        labels, objects = connected_components(mask, 4)
        np.testing.assert_array_equal(labels, mask.astype(np.int32))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['area_px'], 5)
        self.assertEqual(objects[0]['x_max'], 3)
        self.assertEqual(objects[0]['y_max'], 2)
        self.assertAlmostEqual(objects[0]['centroid_y'], .6)

    def test_filter_and_relabel(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[0, 0] = True
        mask[2:4, 2:4] = True
        mask[5, 5:7] = True
        labels, objects = connected_components(mask, min_area=2, max_area=4, exclude_border=True)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['id'], 1)
        self.assertEqual(objects[0]['area_px'], 4)
        self.assertFalse(objects[0]['touches_border'])
        self.assertEqual(int(np.sum(labels > 0)), 4)
        _, objects = connected_components(mask, min_area=1, max_area=2)
        self.assertEqual([item['area_px'] for item in objects], [1, 2])

    def test_empty_and_full(self):
        labels, objects = connected_components(np.zeros((3, 3), dtype=bool))
        self.assertEqual(objects, [])
        self.assertFalse(labels.any())
        _, objects = connected_components(np.ones((3, 3), dtype=bool))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['area_px'], 9)
        self.assertTrue(objects[0]['touches_border'])


if __name__ == '__main__':
    unittest.main()
