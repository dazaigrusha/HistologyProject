"""Малі ручні еталони LBP, гістограм та областей аналізу."""
import unittest
import numpy as np
from histology.texture import analyze_texture, describe_histogram, local_binary_patterns, rgb_to_gray


class TextureTests(unittest.TestCase):
    """Перевірити числа, а не лише форму виходу."""

    def setUp(self):
        self.settings = {'source': 'gray', 'scope': 'whole'}

    def test_manual_code_77(self):
        image = np.array([[9, 2, 8], [1, 5, 7], [6, 3, 4]], dtype=np.uint8)
        codes, valid = local_binary_patterns(image)
        expected = np.zeros((3, 3), dtype=np.uint8); expected[1, 1] = 77
        np.testing.assert_array_equal(codes, expected)
        self.assertEqual(valid.sum(), 1)

    def test_equal_neighbors_and_excluded_border(self):
        result = analyze_texture(np.full((5, 5), 12, dtype=np.uint8), self.settings)
        self.assertEqual(result['histogram_counts'][255], 9)
        self.assertEqual(result['features']['valid_centers'], 9)
        self.assertEqual(result['features']['entropy_bits'], 0)
        self.assertEqual(result['features']['energy'], 1)
        self.assertEqual(result['histogram_probabilities'].sum(), 1)
        self.assertFalse(result['valid_mask'][0].any())

    def test_true_code_zero_is_valid(self):
        image = np.zeros((3, 3), dtype=np.uint8); image[1, 1] = 9
        result = analyze_texture(image, self.settings)
        self.assertEqual(result['histogram_counts'][0], 1)
        self.assertTrue(result['valid_mask'][1, 1])

    def test_tiny_and_empty_scope(self):
        result = analyze_texture(np.zeros((2, 4), dtype=np.uint8), self.settings)
        self.assertEqual(result['features']['valid_centers'], 0)
        self.assertIsNone(result['features']['entropy_bits'])
        self.assertEqual(result['histogram_probabilities'].sum(), 0)
        self.settings['scope'] = 'objects'
        result = analyze_texture(np.zeros((5, 5), dtype=np.uint8), self.settings,
                                 np.zeros((5, 5), dtype=np.int32))
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['features']['valid_centers'], 0)

    def test_object_histograms_and_border_only_object(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1, 1] = 1; labels[2, 2:4] = 2; labels[0, 0] = 3
        self.settings['scope'] = 'objects'
        result = analyze_texture(np.zeros((5, 5), dtype=np.uint8), self.settings, labels)
        self.assertEqual(result['features']['valid_centers'], 3)
        self.assertEqual([item['valid_centers'] for item in result['objects']], [1, 2, 0])
        self.assertIsNone(result['objects'][2]['entropy_bits'])
        self.assertEqual(result['objects'][1]['histogram_counts'][255], 2)

    def test_strictly_increasing_transform(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        first, _ = local_binary_patterns(image)
        second, _ = local_binary_patterns(image * 2 + 3)
        np.testing.assert_array_equal(first, second)

    def test_features_two_codes(self):
        counts = np.zeros(256, dtype=int); counts[0] = counts[255] = 1
        result = describe_histogram(counts)
        self.assertEqual(result['entropy_bits'], 1)
        self.assertEqual(result['energy'], .5)
        self.assertEqual(result['dominant_code'], 0)

    def test_manual_gray(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(rgb_to_gray(rgb), [[76, 150, 29]])

    def test_invalid_scope_and_input(self):
        with self.assertRaises(ValueError):
            local_binary_patterns(np.zeros((3, 3), dtype=float))
        with self.assertRaises(ValueError):
            analyze_texture(np.zeros((3, 3), dtype=np.uint8), {'source': 'gray', 'scope': 'objects'})


if __name__ == '__main__':
    unittest.main()
