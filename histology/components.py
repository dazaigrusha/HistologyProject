"""Ручне маркування зв’язних компонентів через рядкові серії й union-find."""
import numpy as np


def connected_components(mask: np.ndarray, connectivity: int = 8,
                         min_area: int = 1, max_area: int = 0,
                         exclude_border: bool = False) -> tuple:
    """Повернути int32-мітки та список об’єктів після відбору за площею.

    Сусідні серії переднього плану в поточному й попередньому рядках
    об’єднуються, якщо перекриваються (4-зв’язність) або також торкаються
    по діагоналі (8-зв’язність). Після об’єднань мітки ущільнюються до 1..N.
    bbox: x_min,y_min,x_max,y_max; верхні межі виключні. Центроїд — середнє
    координат індексів пікселів (від нуля). max_area=0 вимикає верхню межу.
    """
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
            or mask.ndim != 2 or mask.size == 0):
        raise ValueError('Компоненти очікують непорожній 2D-масив bool.')
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError('Зв’язність має бути 4 або 8.')
    if (type(min_area) is not int or min_area < 1 or type(max_area) is not int
            or max_area < 0 or (max_area and max_area < min_area)
            or type(exclude_border) is not bool):
        raise ValueError('Некоректні параметри відбору компонентів.')
    parent = [0]
    runs = []

    def find(label):
        """Знайти корінь зі стисненням шляху."""
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    previous = []
    margin = 1 if connectivity == 8 else 0
    for y, row in enumerate(mask):
        # Пари [start,stop) визначаємо за переходами 0→1 та 1→0.
        transitions = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        current = []
        cursor = 0
        for start, stop in transitions.reshape(-1, 2):
            start, stop = int(start), int(stop)
            label = len(parent)
            parent.append(label)
            while cursor < len(previous) and previous[cursor][1] + margin <= start:
                cursor += 1
            index = cursor
            while index < len(previous) and previous[index][0] < stop + margin:
                other_root = find(previous[index][2])
                root = find(label)
                if root != other_root:
                    parent[max(root, other_root)] = min(root, other_root)
                index += 1
            current.append((start, stop, label))
            runs.append((y, start, stop, label))
        previous = current
    statistics = {}
    for y, start, stop, label in runs:
        root = find(label)
        count = stop - start
        if root not in statistics:
            statistics[root] = [0, start, y, stop, y + 1, 0.0, 0.0]
        data = statistics[root]
        data[0] += count
        data[1] = min(data[1], start)
        data[2] = min(data[2], y)
        data[3] = max(data[3], stop)
        data[4] = max(data[4], y + 1)
        data[5] += (start + stop - 1) * count / 2
        data[6] += y * count
    height, width = mask.shape
    root_to_id = {}
    objects = []
    for root, (area, x0, y0, x1, y1, sum_x, sum_y) in statistics.items():
        touches_border = x0 == 0 or y0 == 0 or x1 == width or y1 == height
        if area < min_area or (max_area and area > max_area):
            continue
        if exclude_border and touches_border:
            continue
        object_id = len(objects) + 1
        root_to_id[root] = object_id
        objects.append({'id': object_id, 'area_px': area, 'x_min': x0, 'y_min': y0,
                        'x_max': x1, 'y_max': y1, 'centroid_x': sum_x / area,
                        'centroid_y': sum_y / area, 'touches_border': touches_border})
    labels = np.zeros(mask.shape, dtype=np.int32)
    for y, start, stop, label in runs:
        labels[y, start:stop] = root_to_id.get(find(label), 0)
    return labels, objects
