"""Ручний маркерний Watershed на від’ємній евклідовій карті відстаней."""
import heapq
import math
import numpy as np
from histology.components import connected_components


def validate_watershed(settings: dict) -> None:
    """Перевірити параметри автоматичних маркерів."""
    if not isinstance(settings, dict):
        raise ValueError('watershed має бути JSON-об’єктом.')
    bounds = {'min_peak_distance': (1, 1000), 'min_peak_height': (0, 1000),
              'relative_peak_height': (0, 1)}
    for name, (low, high) in bounds.items():
        value = settings.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(f'{name}: потрібне число від {low} до {high}.')


def _squared_distance_1d(values: np.ndarray) -> np.ndarray:
    """Мінімізувати (x-q)²+values[q] нижньою оболонкою парабол.

    Власна реалізація сепарабельного EDT за Felzenszwalb–Huttenlocher.
    Нескінченні значення не утворюють парабол; перетини — у float64.
    """
    sites = np.flatnonzero(np.isfinite(values))
    if sites.size == 0:
        return np.full(values.size, np.inf)
    if np.all(values == 0):
        return np.zeros_like(values)
    vertices = np.empty(sites.size, dtype=np.int64)
    borders = np.empty(sites.size + 1, dtype=np.float64)
    k = 0
    vertices[0] = sites[0]
    borders[0], borders[1] = -np.inf, np.inf
    for site in sites[1:]:
        q = int(site)
        while True:
            previous = int(vertices[k])
            cross = ((values[q] + q * q) - (values[previous] + previous * previous)) / (2 * (q - previous))
            if cross > borders[k]:
                break
            k -= 1
        k += 1
        vertices[k] = q
        borders[k], borders[k + 1] = cross, np.inf
    x = np.arange(values.size)
    # Це вибір інтервалу оболонки, не бібліотечний алгоритм обробки зображень.
    segment = np.searchsorted(borders[1:k + 1], x, side='left')
    nearest = vertices[segment]
    return (x - nearest) ** 2 + values[nearest]


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Точна евклідова відстань до центрів найближчих фонових пікселів.

    Додаткова рамка False задає фон за межами кадру. Фонові значення=0.
    Для одного ізольованого пікселя відстань дорівнює 1, а не 0.5.
    """
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
            or mask.ndim != 2 or mask.size == 0):
        raise ValueError('EDT очікує непорожню 2D-маску bool.')
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)
    padded = np.pad(mask, 1, constant_values=False)
    squared = np.where(padded, np.inf, 0.0)
    for y in range(squared.shape[0]):
        squared[y] = _squared_distance_1d(squared[y])
    for x in range(squared.shape[1]):
        squared[:, x] = _squared_distance_1d(squared[:, x])
    return np.sqrt(squared[1:-1, 1:-1]).astype(np.float32)


def automatic_markers(distance: np.ndarray, components: np.ndarray,
                      settings: dict, connectivity: int = 8) -> np.ndarray:
    """Локальні піки 3×3 → один представник плато → рознесення маркерів.

    Пороги висоти абсолютний і відносний до максимуму своєї компоненти.
    Занадто близькі піки всередині тієї самої компоненти відкидаються.
    Для кожної компоненти завжди лишається хоча б один маркер.
    Рівні кандидати упорядковані за (y,x), випадковості немає.
    """
    validate_watershed(settings)
    if (distance.shape != components.shape or distance.ndim != 2 or distance.size == 0
            or not np.all(np.isfinite(distance)) or np.any(distance < 0)
            or not np.issubdtype(components.dtype, np.integer) or np.any(components < 0)
            or np.any((components > 0) & (distance <= 0))):
        raise ValueError('Некоректні карта відстаней або компоненти.')
    foreground = components > 0
    markers = np.zeros(components.shape, dtype=np.int32)
    if not np.any(foreground):
        return markers
    maxima = np.zeros(int(components.max()) + 1, dtype=np.float32)
    np.maximum.at(maxima, components.ravel(), distance.ravel())
    height, width = distance.shape
    padded = np.pad(distance, 1, constant_values=0)
    peaks = foreground.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            peaks &= distance >= padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
    peaks &= distance >= settings['min_peak_height']
    peaks &= distance >= settings['relative_peak_height'] * maxima[components]
    # Рівні сусідні піки утворюють плато; одна точка — найближча до центру плато.
    peak_labels, plateaus = connected_components(peaks, connectivity=connectivity)
    candidates = []
    for item in plateaus:
        y0, y1, x0, x1 = item['y_min'], item['y_max'], item['x_min'], item['x_max']
        yy, xx = np.nonzero(peak_labels[y0:y1, x0:x1] == item['id'])
        yy, xx = yy + y0, xx + x0
        offsets = (yy - item['centroid_y']) ** 2 + (xx - item['centroid_x']) ** 2
        index = int(np.argmin(offsets))
        y, x = int(yy[index]), int(xx[index])
        candidates.append((-float(distance[y, x]), y, x, int(components[y, x])))
    candidates.sort()
    spacing = settings['min_peak_distance']
    cells = {}
    represented = set()
    next_id = 0
    for _, y, x, component in candidates:
        cy, cx = int(y // spacing), int(x // spacing)
        nearby = [point for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  for point in cells.get((component, cy + dy, cx + dx), [])]
        if any((y - py) ** 2 + (x - px) ** 2 < spacing * spacing for py, px in nearby):
            continue
        next_id += 1
        markers[y, x] = next_id
        cells.setdefault((component, cy, cx), []).append((y, x))
        represented.add(component)
    # Відсутній пік вище порога не має призводити до втрати компоненти.
    missing = np.flatnonzero(maxima > 0)
    missing = [int(value) for value in missing if value > 0 and value not in represented]
    if missing:
        ys, xs = np.nonzero(foreground & (distance == maxima[components]))
        for y, x in zip(ys, xs):
            component = int(components[y, x])
            if component in represented:
                continue
            next_id += 1
            markers[y, x] = next_id
            represented.add(component)
    return markers


def watershed_flood(distance: np.ndarray, markers: np.ndarray,
                    mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Маркерне затоплення рельєфу -distance пріоритетною чергою.

    Рівень шляху=max(попередній рівень,-distance[next]). За рівних
    пріоритетів FIFO урівноважує розростання на плато. Мітка фіксується
    при додаванні пікселя в чергу; задані маркери не змінюються.
    Це повне розбиття: нульових вододільних смуг усередині маски немає,
    межі проходять між сусідніми пікселями з різними ID.
    """
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2
            or mask.size == 0 or distance.shape != mask.shape or markers.shape != mask.shape
            or not np.issubdtype(markers.dtype, np.integer) or np.any(markers < 0)
            or not np.all(np.isfinite(distance)) or np.any(distance < 0)
            or np.any((markers > 0) & ~mask)):
        raise ValueError('Некоректні дані для Watershed.')
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError('Зв’язність має бути 4 або 8.')
    labels = markers.astype(np.int32, copy=True)
    queue = []
    order = 0
    for y, x in zip(*np.nonzero(markers)):
        heapq.heappush(queue, (-float(distance[y, x]), order, int(y), int(x)))
        order += 1
    neighbors = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)]
    height, width = mask.shape
    while queue:
        level, _, y, x = heapq.heappop(queue)
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not mask[ny, nx] or labels[ny, nx] != 0:
                continue
            labels[ny, nx] = labels[y, x]
            heapq.heappush(queue, (max(level, -float(distance[ny, nx])), order, ny, nx))
            order += 1
    if np.any(mask & (labels == 0)):
        raise ValueError('Кожна зв’язна компонента маски повинна мати маркер.')
    return labels


def split_touching(mask: np.ndarray, settings: dict, connectivity: int = 8) -> dict:
    """Автоматично обчислити EDT, маркери та Watershed-мітки на всій масці."""
    validate_watershed(settings)
    components, _ = connected_components(mask, connectivity=connectivity)
    distance = distance_transform(mask)
    markers = automatic_markers(distance, components, settings, connectivity)
    labels = watershed_flood(distance, markers, mask, connectivity)
    return {'distance': distance, 'markers': markers, 'watershed_labels': labels}
