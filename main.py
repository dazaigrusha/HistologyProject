"""Запуск вікна або вибраного методу без графічного інтерфейсу."""
import argparse
from pathlib import Path
from PIL import Image
from histology.config import load_config


def main():
    """Завантажити конфігурацію та виконати запитаний сценарій."""
    parser = argparse.ArgumentParser(description="Аналіз гістологічних зображень")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json')
    parser.add_argument('--input', type=Path, help='Вхідний файл для запуску без вікна')
    parser.add_argument('--method', choices=('stains', 'otsu', 'opening', 'watershed', 'lbp', 'all'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if (args.input is None) != (args.method is None):
        parser.error('--input та --method потрібно вказувати разом.')
    try:
        config = load_config(args.config)
        if args.method is not None:
            from histology.cli import run_method
            destination = run_method(args.input, args.method, args.output, config)
            print(f'Результати збережено: {destination}')
            return
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        parser.exit(1, f'Помилка: {error}\n')
    from histology.app import Application
    Application(config).mainloop()


if __name__ == '__main__':
    main()
