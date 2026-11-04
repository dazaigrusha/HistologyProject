"""Площі, периметри, округлість і щільність маркованих кандидатів."""
import math
import numpy as np


def validate_measurements(settings: dict) -> None:
    """Перевірити необов’язкову калібровку квадратного пікселя."""
    if not isinstance(settings, dict) or 'pixel_size_um' not in settings:
        raise ValueError('measurements має містити pixel_size_um (число або null).')
    value = settings['pixel_size_um']
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                              or not 0 < value <= 10000):
        raise ValueError('pixel_size_um: додатне число до 10000 або null.')


def measure_regions(labels: np.ndarray, pixel_size_um: float | None = None) -> list:
    """Повернути статистику кожної додатної мітки (без зміни сегментації).

    Периметр — кількість відкритих сторін квадратних пікселів (4-сусідня
    сітка), включно з отворами й межами між мітками. Округлість=4πA/P²;
    вона має сіткове зміщення, це не субпіксельний контурний периметр.
    Калібровка припускає однакові масштаби по X і Y.
    """
    validate_measurements({'pixel_size_um': pixel_size_um})
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.size == 0
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError('Очікуються непорожні невід’ємні цілочислові мітки.')
    maximum = int(labels.max())
    count = np.bincount(labels.ravel(), minlength=maximum + 1)
    perimeter = np.zeros(maximum + 1, dtype=np.int64)
    padded = np.pad(labels, 1)
    height, width = labels.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        changed = labels[(labels > 0) & (labels != other)]
        perimeter += np.bincount(changed, minlength=maximum + 1)
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    x_min, y_min = np.full(maximum + 1, width), np.full(maximum + 1, height)
    x_max, y_max = np.zeros(maximum + 1, dtype=int), np.zeros(maximum + 1, dtype=int)
    np.minimum.at(x_min, ids, xs)
    np.minimum.at(y_min, ids, ys)
    np.maximum.at(x_max, ids, xs + 1)
    np.maximum.at(y_max, ids, ys + 1)
    sum_x = np.bincount(ids, weights=xs, minlength=maximum + 1)
    sum_y = np.bincount(ids, weights=ys, minlength=maximum + 1)
    objects = []
    for object_id in np.flatnonzero(count[1:]) + 1:
        area = int(count[object_id])
        length = int(perimeter[object_id])
        diameter = math.sqrt(4 * area / math.pi)
        item = {'id': int(object_id), 'area_px': area,
                'x_min': int(x_min[object_id]), 'y_min': int(y_min[object_id]),
                'x_max': int(x_max[object_id]), 'y_max': int(y_max[object_id]),
                'centroid_x': float(sum_x[object_id] / area),
                'centroid_y': float(sum_y[object_id] / area),
                'touches_border': bool(x_min[object_id] == 0 or y_min[object_id] == 0
                                       or x_max[object_id] == width or y_max[object_id] == height),
                'perimeter_px': length, 'circularity': 4 * math.pi * area / (length * length),
                'equivalent_diameter_px': diameter,
                'area_um2': None if pixel_size_um is None else area * pixel_size_um ** 2,
                'perimeter_um': None if pixel_size_um is None else length * pixel_size_um,
                'equivalent_diameter_um': None if pixel_size_um is None else diameter * pixel_size_um}
        objects.append(item)
    return objects


def filter_regions(labels: np.ndarray, settings: dict,
                   pixel_size_um: float | None = None) -> tuple:
    """Відфільтрувати Watershed-басейни за площею/краєм та ущільнити ID.

    На відміну від повторного connected-components, зберігає розділення
    суміжних басейнів з різними мітками.
    """
    from histology.segmentation import validate_segmentation
    validate_segmentation(settings)
    objects = measure_regions(labels, pixel_size_um)
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    selected = []
    for item in objects:
        if item['area_px'] < settings['min_area']:
            continue
        if settings['max_area'] and item['area_px'] > settings['max_area']:
            continue
        if settings['exclude_border'] and item['touches_border']:
            continue
        new_id = len(selected) + 1
        lookup[item['id']] = new_id
        item['id'] = new_id
        selected.append(item)
    return lookup[labels], selected


def summarize_regions(objects: list, shape: tuple,
                      pixel_size_um: float | None = None) -> dict:
    """Щільність на всю площу кадру; порожня вибірка дає null для середніх."""
    validate_measurements({'pixel_size_um': pixel_size_um})
    pixels = int(shape[0]) * int(shape[1])
    count = len(objects)
    return {'candidate_count': count, 'frame_area_px': pixels,
            'foreground_area_px': sum(item['area_px'] for item in objects),
            'density_per_megapixel': count * 1000000 / pixels,
            'density_per_mm2': None if pixel_size_um is None else count * 1000000 / (pixels * pixel_size_um ** 2),
            'mean_area_px': None if not objects else sum(item['area_px'] for item in objects) / count,
            'mean_circularity': None if not objects else sum(item['circularity'] for item in objects) / count,
            'density_reference': 'whole_image_frame', 'pixel_size_um': pixel_size_um}
