"""Послідовний запуск ручної сегментації та підготовка візуалізації."""
import numpy as np
from PIL import Image, ImageDraw
from histology.components import connected_components
from histology.segmentation import binary_opening, otsu_threshold, validate_segmentation
from histology.stains import transmittance_image


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
