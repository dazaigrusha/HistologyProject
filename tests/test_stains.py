"""Малі еталони з ручним розрахунком, без готових CV-реалізацій."""
import copy
import unittest
import numpy as np
from histology.stains import separate_stains, transmittance_image, validate_stains


class StainTests(unittest.TestCase):
    """Перевірки формули OD, системи рівнянь і граничних випадків."""

    def setUp(self):
        self.settings = {'hematoxylin': [1, 0, 0], 'eosin': [0, 1, 0],
                         'white_level': 255, 'block_rows': 1}

    def test_hand_computed_3_by_3(self):
        """Ортогональні вектори: H=-ln(R/255), E=-ln(G/255)."""
        rgb = np.full((3, 3, 3), 255, dtype=np.uint8)
        rgb[0, 0] = [85, 255, 255]  # H=ln(3), E=0.
        rgb[1, 1] = [255, 51, 255]  # H=0, E=ln(5).
        h_map, e_map = separate_stains(rgb, self.settings)
        expected_h = np.zeros((3, 3)); expected_h[0, 0] = np.log(3)
        expected_e = np.zeros((3, 3)); expected_e[1, 1] = np.log(5)
        np.testing.assert_allclose(h_map, expected_h, atol=1e-6)
        np.testing.assert_allclose(e_map, expected_e, atol=1e-6)
        self.assertEqual(int(transmittance_image(h_map)[0, 0]), 85)
        self.assertEqual(int(transmittance_image(e_map)[1, 1]), 51)

    def test_oblique_vectors_manual_solution(self):
        """h=(1,0,0), e=(.6,.8,0); OD=(ln3,ln5,0).

        cE=ln5/.8; cH=ln3-.75*ln5<0, тому після обрізання cH=0.
        """
        self.settings['eosin'] = [.6, .8, 0]
        h_map, e_map = separate_stains(np.array([[[85, 51, 255]]], dtype=np.uint8),
                                     self.settings)
        self.assertEqual(float(h_map[0, 0]), 0)
        self.assertAlmostEqual(float(e_map[0, 0]), np.log(5) / .8, places=6)

    def test_black_white_and_blocks(self):
        rgb = np.array([[[0, 0, 0]], [[255, 255, 255]]], dtype=np.uint8)
        h_map, e_map = separate_stains(rgb, self.settings)
        np.testing.assert_allclose(h_map[:, 0], [np.log(255), 0], atol=1e-6)
        self.settings['block_rows'] = 128
        h_again, e_again = separate_stains(rgb, self.settings)
        np.testing.assert_array_equal(h_map, h_again)
        np.testing.assert_array_equal(e_map, e_again)

    def test_invalid_vectors(self):
        for invalid in ([1, 0, 0], [0, 0, 0], [float('nan'), 0, 0], [True, 0, 0]):
            settings = copy.deepcopy(self.settings)
            settings['eosin'] = invalid
            with self.assertRaises(ValueError):
                validate_stains(settings)

    def test_wrong_input(self):
        for invalid in (np.zeros((3, 3)), np.zeros((0, 2, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                separate_stains(invalid, self.settings)


if __name__ == '__main__':
    unittest.main()
