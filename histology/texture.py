"""Ручний LBP 3×3, гістограми та прості описові ознаки текстури."""
import numpy as np
from PIL import Image, ImageDraw

# Біти за годинниковою стрілкою, починаючи з верхнього лівого сусіда.
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
           (1, 1), (1, 0), (1, -1), (0, -1))


def validate_texture(settings: dict) -> None:
    """Перевірити джерело та область підрахунку ознак."""
    if not isinstance(settings, dict):
        raise ValueError('texture має бути JSON-об’єктом.')
    if settings.get('source') not in ('hematoxylin', 'gray'):
        raise ValueError('texture.source: hematoxylin або gray.')
    if settings.get('scope') not in ('whole', 'objects'):
        raise ValueError('texture.scope: whole або objects.')


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Ручна яскравість RGB: round(.299R+.587G+.114B), uint8."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Очікується RGB uint8.')
    return np.rint(np.sum(rgb.astype(np.float64) * [.299, .587, .114], axis=2)).astype(np.uint8)


def local_binary_patterns(gray: np.ndarray) -> tuple:
    """Повернути коди uint8 і маску допустимих центрів bool.

    Для кожного сусіда bit=1, якщо neighbor>=center, інакше 0.
    code=sum(bit[k]*2**k). Межі завширшки 1 не обчислюються, їхні коди=0.
    Для кадрів менших 3×3 усі центри недопустимі; це не помилка.
    """
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8
            or gray.ndim != 2 or gray.size == 0):
        raise ValueError('LBP очікує непорожній 2D-масив uint8.')
    codes = np.zeros_like(gray)
    valid = np.zeros(gray.shape, dtype=bool)
    if min(gray.shape) < 3:
        return codes, valid
    height, width = gray.shape
    center = gray[1:-1, 1:-1]
    for bit, (dy, dx) in enumerate(OFFSETS):
        neighbor = gray[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
        codes[1:-1, 1:-1] |= (neighbor >= center).astype(np.uint8) << bit
    valid[1:-1, 1:-1] = True
    return codes, valid


def describe_histogram(counts: np.ndarray) -> dict:
    """Ентропія та енергія розподілу; порожня область має null-ознаки."""
    total = int(counts.sum())
    if total == 0:
        return {'valid_centers': 0, 'entropy_bits': None, 'energy': None,
                'dominant_code': None}
    probabilities = counts.astype(np.float64) / total
    nonzero = probabilities[probabilities > 0]
    return {'valid_centers': total,
            'entropy_bits': float(-np.sum(nonzero * np.log2(nonzero))),
            'energy': float(np.sum(probabilities * probabilities)),
            'dominant_code': int(np.argmax(counts))}


def analyze_texture(gray: np.ndarray, settings: dict,
                    labels: np.ndarray | None = None) -> dict:
    """LBP по всьому полю або центрах об’єктів; також гістограми кожного ID.

    Вікно може включати фон поза об’єктом; scope визначає лише його центр.
    labels мають походити з того самого кадру й поточної стадії сегментації.
    """
    validate_texture(settings)
    codes, valid = local_binary_patterns(gray)
    if labels is not None and (labels.shape != gray.shape
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError('Мітки LBP повинні відповідати розміру зображення.')
    if settings['scope'] == 'objects':
        if labels is None:
            raise ValueError('Для texture.scope=objects спочатку знайди кандидатів.')
        valid &= labels > 0
    counts = np.zeros(256, dtype=np.int64)
    np.add.at(counts, codes[valid], 1)
    total = int(counts.sum())
    probabilities = counts.astype(np.float64) / total if total else np.zeros(256)
    objects = []
    # Один прохід накопичує гістограми всіх ID без повторного сканування кадру.
    if labels is not None:
        ids = np.unique(labels[labels > 0])
        positions = np.searchsorted(ids, labels[valid])
        has_object = labels[valid] > 0
        packed = positions[has_object] * 256 + codes[valid][has_object].astype(np.int64)
        all_counts = np.bincount(packed, minlength=len(ids) * 256).reshape(len(ids), 256)
        for object_id, histogram in zip(ids, all_counts):
            objects.append({'id': int(object_id), **describe_histogram(histogram),
                            'histogram_counts': histogram.tolist()})
    return {'codes': codes, 'valid_mask': valid, 'histogram_counts': counts,
            'histogram_probabilities': probabilities, 'features': describe_histogram(counts),
            'objects': objects, 'settings': settings.copy()}


def histogram_image(counts: np.ndarray) -> Image.Image:
    """Намалювати всі 256 частот Pillow; це лише візуалізація готових чисел."""
    width, height = 900, 320
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 65, 45, 875, 270
    maximum = max(1, int(counts.max()))
    draw.text((left, 12), f'LBP histogram | centers: {int(counts.sum())} | max count: {int(counts.max())}', fill='black')
    draw.line((left, top, left, bottom, right, bottom), fill='black')
    for code, count in enumerate(counts):
        x0 = left + code * (right - left) / 256
        x1 = left + (code + 1) * (right - left) / 256
        y = bottom - int(count) * (bottom - top) / maximum
        if count:
            draw.rectangle((int(x0), int(y), max(int(x0), int(x1) - 1), bottom - 1), fill='#7654a3')
    for code in (0, 64, 128, 192, 255):
        draw.text((left + code * (right - left) / 256 - 6, bottom + 8), str(code), fill='black')
    draw.text((left - 20, top - 8), str(int(counts.max())), fill='black')
    draw.text((left - 15, bottom), '0', fill='black')
    draw.text((400, 300), 'LBP code (0-255); frequency', fill='black')
    return image
