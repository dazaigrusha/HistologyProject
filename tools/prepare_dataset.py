"""Імпортувати локальні tissue images + instance label masks без втрати ID."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from histology.dataset import prepare_dataset


def main():
    """Створити manifests calibration/test для batch-команди."""
    parser = argparse.ArgumentParser(description='Підготовка гістологічної вибірки')
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/prepared'))
    parser.add_argument('--mask-suffix', default='')
    parser.add_argument('--group-regex')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        folder = prepare_dataset(args.images, args.labels, args.output,
            mask_suffix=args.mask_suffix, group_regex=args.group_regex, seed=args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Помилка імпорту: {error}\n')
    print(f'Вибірка підготовлена: {folder}')
    print('Перевір dataset.json, підбирай параметри на calibration.csv, оцінюй на test.csv.')


if __name__ == '__main__':
    main()
