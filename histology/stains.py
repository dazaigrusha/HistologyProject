"""Ручне двокомпонентне розділення H&E в просторі оптичної густини.

Теоретична основа: Ruifrok & Johnston (2001), Landini et al. (2021).
Використовуємо явний розв'язок системи 2×2, а не готовий CV-алгоритм.
"""
import math
import numpy as np


def validate_stains(settings: dict) -> tuple:
    """Перевірити параметри й повернути одиничні вектори H, E та їх добуток."""
    if not isinstance(settings, dict):
        raise ValueError('stains має бути JSON-об’єктом.')
    vectors = []
    for name in ('hematoxylin', 'eosin'):
        raw = settings.get(name)
        if (not isinstance(raw, list) or len(raw) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       or not 0 <= x <= 1 for x in raw)):
            raise ValueError(f'{name}: потрібні три скінченні числа від 0 до 1.')
        vector = np.asarray(raw, dtype=np.float64)
        length = math.sqrt(float(np.sum(vector * vector)))
        if length == 0:
            raise ValueError(f'{name}: вектор не може бути нульовим.')
        vectors.append(vector / length)
    h_vector, e_vector = vectors
    dot = float(np.sum(h_vector * e_vector))
    if 1 - dot * dot < 1e-4:
        raise ValueError('Вектори H та E надто близькі: розділення нестійке.')
    white = settings.get('white_level')
    if type(white) not in (int, float) or not math.isfinite(white) or not 1 <= white <= 255:
        raise ValueError('white_level: потрібне число від 1 до 255.')
    rows = settings.get('block_rows')
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError('block_rows: потрібне ціле число від 1 до 4096.')
    return h_vector, e_vector, dot


def separate_stains(rgb: np.ndarray, settings: dict) -> tuple:
    """Повернути карти оптичної густини H та E (float32) без масштабування.

    OD = -ln(max(I, 1) / I0); для I > I0 густина обмежується нулем.
    Для одиничних векторів h та e розв'язуємо нормальні рівняння:
    cH = (OD·h - d * OD·e)/(1-d²), cE = (OD·e - d * OD·h)/(1-d²),
    де d = h·e. Від'ємні оцінки після розв'язання обрізаються до нуля.
    Це проєкція методом найменших квадратів із постобмеженням,
    а не повний невід'ємний least-squares. Пікселі обробляються смугами.
    """
    h_vector, e_vector, dot = validate_stains(settings)
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0):
        raise ValueError('Очікується непорожній RGB-масив uint8 розміру H×W×3.')
    h_map = np.empty(rgb.shape[:2], dtype=np.float32)
    e_map = np.empty_like(h_map)
    denominator = 1 - dot * dot
    rows = settings['block_rows']
    for start in range(0, rgb.shape[0], rows):
        stop = min(start + rows, rgb.shape[0])
        intensity = rgb[start:stop].astype(np.float64)
        od = -np.log(np.maximum(intensity, 1) / settings['white_level'])
        od = np.maximum(od, 0)
        h_projection = np.sum(od * h_vector, axis=2)
        e_projection = np.sum(od * e_vector, axis=2)
        h_map[start:stop] = np.maximum((h_projection - dot * e_projection) / denominator, 0)
        e_map[start:stop] = np.maximum((e_projection - dot * h_projection) / denominator, 0)
    return h_map, e_map


def transmittance_image(concentration: np.ndarray) -> np.ndarray:
    """Перевести карту OD у uint8: T=255·exp(-c); сильніше забарвлення темніше.

    Фіксоване перетворення без min-max нормалізації між зображеннями.
    Для наступної сегментації зберігаємо вихідні float-карти окремо.
    """
    return np.rint(255 * np.exp(-np.maximum(concentration, 0))).astype(np.uint8)
