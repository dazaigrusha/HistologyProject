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
    parser.add_argument('--batch', type=Path, help='CSV вибірки зі стовпцями image,reference')
    parser.add_argument('--reference', type=Path, help='Еталон для одного CLI-запуску')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.batch is not None and (args.input is not None or args.reference is not None):
        parser.error('--batch не поєднується з --input або --reference.')
    if args.batch is None and (args.input is None) != (args.method is None):
        parser.error('--input та --method потрібно вказувати разом.')
    if args.reference is not None and args.input is None:
        parser.error('--reference потребує --input.')
    try:
        config = load_config(args.config)
        if args.batch is not None:
            from histology.batch import run_batch
            destination = run_batch(args.batch, args.output, config, args.method or 'all')
            print(f'Пакетний звіт збережено: {destination}')
            return
        if args.method is not None:
            from histology.cli import run_method
            destination = run_method(args.input, args.method, args.output, config)
            if args.reference is not None:
                from histology.evaluation import evaluate_run
                evaluate_run(destination, args.reference, config['max_pixels'])
            print(f'Результати збережено: {destination}')
            return
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        parser.exit(1, f'Помилка: {error}\n')
    from histology.app import Application
    Application(config).mainloop()


if __name__ == '__main__':
    main()
