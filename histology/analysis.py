"""Послідовний запуск ручної сегментації та підготовка візуалізації."""
import numpy as np
from PIL import Image, ImageDraw
from histology.components import connected_components
from histology.segmentation import binary_opening, otsu_threshold, validate_segmentation
from histology.stains import transmittance_image
from histology.measurements import filter_regions, measure_regions, summarize_regions
from histology.watershed import split_touching


def segment_nuclei(h_map: np.ndarray, settings: dict) -> dict:
    """Оцу → відкриття → компоненти; повернути проміжні маски та об’єкти.

    Яскравість strength=255-T_H зростає зі внеском гематоксиліну.
    Відбір виконується на повній роздільності; одиниці площі — пікселі².
    Назва функції позначає ціль: виходи є лише кандидатами в ядра.
    """
    validate_segmentation(settings)
    if (not isinstance(h_map, np.ndarray) or h_map.ndim != 2 or h_map.size == 0
            or not np.issubdtype(h_map.dtype, np.floating)
            or not np.all(np.isfinite(h_map)) or np.any(h_map < 0)):
        raise ValueError('Очікується непорожня скінченна невід’ємна float-карта H.')
    strength = 255 - transmittance_image(h_map)
    otsu = otsu_threshold(strength)
    effective = max(otsu, settings['threshold_floor'])
    raw_mask = strength > effective
    opened_mask = binary_opening(raw_mask, settings['opening_radius'])
    labels, objects = connected_components(
        opened_mask, connectivity=settings['connectivity'], min_area=settings['min_area'],
        max_area=settings['max_area'], exclude_border=settings['exclude_border'])
    return {'strength': strength, 'otsu_threshold': otsu, 'effective_threshold': effective,
            'raw_mask': raw_mask, 'opened_mask': opened_mask, 'mask': labels > 0,
            'labels': labels, 'objects': objects, 'count': len(objects)}


def object_overlay(rgb: np.ndarray, labels: np.ndarray, objects: list) -> Image.Image:
    """Показати зелені межі кандидатів і номери; Pillow лише для візуалізації."""
    padded = np.pad(labels, 1)
    foreground = labels > 0
    border = np.zeros_like(foreground)
    height, width = labels.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        border |= foreground & (neighbor != labels)
    output = rgb.copy()
    output[border] = [0, 255, 80]
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    # Номер ставимо на першому пікселі об’єкта: центроїд може потрапити в отвір.
    for item in objects:
        y = item['y_min']
        row = labels[y, item['x_min']:item['x_max']]
        x = item['x_min'] + int(np.flatnonzero(row == item['id'])[0])
        draw.text((x, y), str(item['id']), fill='yellow', stroke_width=1, stroke_fill='black')
    return image


def apply_watershed(previous: dict, segmentation: dict, watershed: dict,
                    measurements: dict) -> dict:
    """Розділити відкриту маску, зберігши результати до Watershed для порівняння.

    Верхню межу площі й виключення краю застосовуємо після розділення:
    велика злипла область може містити кілька допустимих ядер.
    Відкидаємо перед розділенням лише компоненти менші за min_area.
    Повторний запуск завжди починається з opened_mask, не з попередніх басейнів.
    """
    source_labels, _ = connected_components(previous['opened_mask'],
        connectivity=segmentation['connectivity'], min_area=segmentation['min_area'])
    split = split_touching(source_labels > 0, watershed, segmentation['connectivity'])
    scale = measurements['pixel_size_um']
    labels, objects = filter_regions(split['watershed_labels'], segmentation, scale)
    result = previous.copy()
    result.pop('evaluation', None)  # Після нової сегментації старі метрики неактуальні.
    before_labels = previous.get('before_labels', previous['labels'])
    before_objects = measure_regions(before_labels, scale)
    result.update(split)
    result.update({'before_labels': before_labels, 'before_objects': before_objects,
                   'before_count': len(before_objects), 'labels': labels, 'objects': objects,
                   'count': len(objects), 'mask': labels > 0, 'stage': 'watershed',
                   'marker_count': int(split['markers'].max()),
                   'basin_count_before_filter': int(split['watershed_labels'].max()),
                   'summary': summarize_regions(objects, labels.shape, scale),
                   'before_summary': summarize_regions(before_objects, labels.shape, scale)})
    return result


def add_measurements(result: dict, measurements: dict) -> dict:
    """Додати однакові вимірювання й щільність для порівняння до/після."""
    objects = measure_regions(result['labels'], measurements['pixel_size_um'])
    result.update(objects=objects, stage='components',
                  summary=summarize_regions(objects, result['labels'].shape,
                                            measurements['pixel_size_um']))
    return result


def label_image(labels: np.ndarray) -> Image.Image:
    """Детерміновані кольори ID; чорний=фон, сусідні басейни видимі окремо."""
    ids = np.arange(int(labels.max()) + 1, dtype=np.int64)
    palette = np.column_stack((64 + (ids * 67) % 192,
                               64 + (ids * 101) % 192, 64 + (ids * 137) % 192)).astype(np.uint8)
    palette[0] = 0
    return Image.fromarray(palette[labels])


def distance_image(distance: np.ndarray) -> Image.Image:
    """Лише перегляд EDT: від 0 до максимуму кадру → від 0 до 255."""
    maximum = float(distance.max())
    preview = np.zeros(distance.shape, dtype=np.uint8) if maximum == 0 else np.rint(distance * (255 / maximum)).astype(np.uint8)
    return Image.fromarray(preview)


def marker_image(distance: np.ndarray, markers: np.ndarray) -> Image.Image:
    """Маркери збільшені до кружечків для видимості; числові мітки не змінюємо."""
    image = distance_image(distance).convert('RGB')
    draw = ImageDraw.Draw(image)
    for y, x in zip(*np.nonzero(markers)):
        draw.ellipse((int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2), fill='red')
    return image
