"""Введення зображень; алгоритмів обробки на цьому етапі ще немає."""
from pathlib import Path
import warnings
from PIL import Image, ImageOps


def load_image(path: Path, max_pixels: int) -> Image.Image:
    """Прочитати перший кадр, врахувати EXIF, повернути незалежну RGB-копію.

    Прозорі області показуються на білому фоні. Масштаб для аналізу
    не змінюється. Надвеликі зображення відхиляються до декодування.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as source:
            if source.width * source.height > max_pixels:
                raise ValueError(f'Зображення перевищує ліміт {max_pixels:,} пікселів.')
            source.seek(0)
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            return Image.alpha_composite(background, rgba).convert('RGB')
