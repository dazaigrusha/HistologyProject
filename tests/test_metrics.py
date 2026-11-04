"""Ручні числові еталони перекриття, порожніх масок та кількості."""
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from histology.metrics import binary_metrics, count_metrics, evaluate_result, read_reference


class MetricTests(unittest.TestCase):
    """Помилка кількості не підміняє зіставлення окремих екземплярів."""

    def test_manual_overlap(self):
        predicted = np.array([[1, 1], [0, 0]], dtype=bool)
        reference = np.array([[1, 0], [1, 0]], dtype=bool)
        result = binary_metrics(predicted, reference)
        self.assertEqual([result[key] for key in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])
        self.assertEqual(result['dice'], .5)
        self.assertAlmostEqual(result['iou'], 1 / 3)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['mse_binary'], .5)
        self.assertAlmostEqual(result['psnr_db'], 10 * math.log10(2))
        self.assertAlmostEqual(result['ssim_global'], .0009 / .5009)

    def test_empty_and_perfect_masks(self):
        empty = np.zeros((3, 3), dtype=bool)
        result = binary_metrics(empty, empty)
        self.assertEqual(result['dice'], 1)
        self.assertEqual(result['iou'], 1)
        self.assertEqual(result['ssim_global'], 1)
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['psnr_db'])
        self.assertTrue(result['psnr_infinite'])
        full = ~empty
        result = binary_metrics(full, empty)
        self.assertEqual(result['dice'], 0)
        self.assertEqual(result['mse_binary'], 1)
        self.assertEqual(result['psnr_db'], 0)

    def test_count_and_zero_reference(self):
        self.assertEqual(count_metrics(3, 2)['count_relative_error'], .5)
        self.assertEqual(count_metrics(3, 0)['count_absolute_error'], 3)
        self.assertIsNone(count_metrics(3, 0)['count_relative_error'])
        self.assertIsNone(count_metrics(3, None)['count_absolute_error'])

    def test_reference_ids_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truth.npy'
            np.save(path, np.array([[0, 7], [31, 31]], dtype=np.int32))
            mask, count = read_reference(path, 100)
            self.assertEqual(count, 2)
            self.assertEqual(int(mask.sum()), 3)
            image = Path(tmp) / 'truth.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(image)
            _, count = read_reference(image, 100)
            self.assertIsNone(count)
            np.save(path, np.array([[0., .5]]))
            with self.assertRaises(ValueError):
                read_reference(path, 100)

    def test_mismatched_masks(self):
        with self.assertRaises(ValueError):
            binary_metrics(np.zeros((3, 3), dtype=bool), np.zeros((2, 2), dtype=bool))

    def test_touching_ids_same_overlap_but_different_count_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'instances.npy'
            np.save(path, np.array([[7, 7, 9, 9]], dtype=np.int32))
            result = {'labels': np.array([[1, 1, 2, 2]], dtype=np.int32),
                      'before_labels': np.array([[1, 1, 1, 1]], dtype=np.int32)}
            evaluation = evaluate_result(result, path, 100)
            before, after = evaluation['rows']
            self.assertEqual(before['dice'], 1)
            self.assertEqual(after['dice'], 1)
            self.assertEqual(before['count_absolute_error'], 1)
            self.assertEqual(after['count_absolute_error'], 0)


if __name__ == '__main__':
    unittest.main()
