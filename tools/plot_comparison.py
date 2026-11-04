"""Зберегти науковий графік порівняння зі звіту batch."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from histology.plots import plot_comparison


def main():
    """Прочитати comparison.csv без ручного перенесення чисел."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    try:
        path = plot_comparison(args.batch)
    except ModuleNotFoundError:
        parser.exit(1, 'Встанови додаткові залежності: python -m pip install -r requirements-report.txt\n')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Помилка графіка: {error}\n')
    print(f'Графік збережено: {path}')


if __name__ == '__main__':
    main()
