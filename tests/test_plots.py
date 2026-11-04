"""Побудова графіка без графічного дисплея (додатковий пакет matplotlib)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from histology.plots import plot_comparison


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Install requirements-report.txt')
class PlotTests(unittest.TestCase):
    """Файл створюється лише за наявності реальної таблиці чисел."""

    def test_plot_from_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'comparison.csv').write_text(
                'stage,mean_dice,mean_iou,mean_count_absolute_error,n_count_absolute_error\n'
                'before_watershed,0.7,0.55,5,3\nfinal,0.8,0.67,2,3\n')
            path = plot_comparison(root)
            self.assertGreater(path.stat().st_size, 1000)

    def test_empty_table_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'comparison.csv').write_text('stage,mean_dice\n')
            with self.assertRaises(ValueError):
                plot_comparison(root)


if __name__ == '__main__':
    unittest.main()
