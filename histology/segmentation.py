"""Ручний поріг Оцу та бінарне морфологічне відкриття."""
import numpy as np


def validate_segmentation(settings: dict) -> None:
    """Перевірити параметри маски та відбору компонентів."""
    if not isinstance(settings, dict):
        raise ValueError('segmentation має бути JSON-об’єктом.')
    limits = {'threshold_floor': (0, 254), 'opening_radius': (0, 10),
              'min_area': (1, 1000000000), 'max_area': (0, 1000000000)}
    for name, (low, high) in limits.items():
        value = settings.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name}: потрібне ціле число від {low} до {high}.')
    if settings['max_area'] and settings['max_area'] < settings['min_area']:
        raise ValueError('max_area має бути 0 або не меншим за min_area.')
    if type(settings.get('connectivity')) is not int or settings['connectivity'] not in (4, 8):
        raise ValueError('connectivity: потрібно 4 або 8.')
    if type(settings.get('exclude_border')) is not bool:
        raise ValueError('exclude_border: потрібно true або false.')


def otsu_threshold(gray: np.ndarray) -> int:
    """Повернути поріг для класів [0,t] та [t+1,255].

    Гістограму будуємо накопиченням лічильників. Перебираємо 256 порогів
    і максимізуємо n0*n1*(mean0-mean1)^2. Для рівних максимумів беремо
    найменший поріг. Для однорідного кадру повертаємо його єдиний рівень;
    перевірка gray>t тоді дає порожню маску (немає двох класів).
    """
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8
            or gray.ndim != 2 or gray.size == 0):
        raise ValueError('Оцу очікує непорожній 2D-масив uint8.')
    histogram = np.zeros(256, dtype=np.int64)
    np.add.at(histogram, gray.ravel(), 1)
    total = int(gray.size)
    total_sum = sum(level * int(count) for level, count in enumerate(histogram))
    left_count = left_sum = 0
    best_score = -1.0
    best_threshold = int(gray.flat[0])
    for threshold, count in enumerate(histogram):
        left_count += int(count)
        left_sum += threshold * int(count)
        right_count = total - left_count
        if left_count == 0 or right_count == 0:
            continue
        difference = left_sum / left_count - (total_sum - left_sum) / right_count
        score = left_count * right_count * difference * difference
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def binary_opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Ерозія, потім дилатація диском dx²+dy²<=radius².

    За межами кадру завжди фон (False). radius=0 — копія без змін.
    Немає готових морфологічних функцій: лише зсуви та AND/OR.
    """
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
            or mask.ndim != 2 or mask.size == 0):
        raise ValueError('Морфологія очікує непорожній 2D-масив bool.')
    if type(radius) is not int or not 0 <= radius <= 10:
        raise ValueError('Радіус має бути цілим від 0 до 10.')
    height, width = mask.shape
    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius]
    padded = np.pad(mask, radius, constant_values=False)
    eroded = np.ones_like(mask)
    for dy, dx in offsets:
        eroded &= padded[radius + dy:radius + dy + height,
                         radius + dx:radius + dx + width]
    padded = np.pad(eroded, radius, constant_values=False)
    opened = np.zeros_like(mask)
    for dy, dx in offsets:
        opened |= padded[radius + dy:radius + dy + height,
                         radius + dx:radius + dx + width]
    return opened
