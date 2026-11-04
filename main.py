"""Точка входу навчального проєкту."""
import argparse
from pathlib import Path
from histology.config import load_config


def main():
    """Завантажити конфігурацію та відкрити вікно перегляду."""
    parser = argparse.ArgumentParser(description="Аналіз гістологічних зображень")
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent / 'config.json')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Помилка конфігурації: {error}\n')
    from histology.app import Application
    Application(config).mainloop()


if __name__ == '__main__':
    main()
