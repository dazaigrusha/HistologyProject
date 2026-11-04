"""Спільні сценарії обчислень для вікна й командного рядка."""
import numpy as np
from histology.analysis import add_measurements, apply_watershed, object_overlay, segment_nuclei
from histology.stains import transmittance_image
from histology.texture import analyze_texture, rgb_to_gray


def analyze_image(rgb: np.ndarray, h_map: np.ndarray, settings: dict, measurements: dict) -> dict:
    """Маска та накладання з вимірюваннями."""
    result = add_measurements(segment_nuclei(h_map, settings), measurements)
    result['overlay'] = object_overlay(rgb, result['labels'], result['objects'])
    return result


def split_image(rgb: np.ndarray, previous: dict, config: dict) -> dict:
    """Watershed з накладаннями до й після розділення."""
    result = apply_watershed(previous, config['segmentation'], config['watershed'], config['measurements'])
    result['before_overlay'] = object_overlay(rgb, result['before_labels'], result['before_objects'])
    result['overlay'] = object_overlay(rgb, result['labels'], result['objects'])
    return result


def texture_image(rgb: np.ndarray, maps: tuple | None, result: dict | None,
                  settings: dict) -> dict:
    """Вибрати джерело LBP та поточні мітки; без залежності від Tkinter."""
    if settings['source'] == 'hematoxylin':
        if maps is None:
            raise ValueError('Спочатку розділи H&E для LBP гематоксилінового каналу.')
        gray = 255 - transmittance_image(maps[0])
    else:
        gray = rgb_to_gray(rgb)
    labels = None if result is None else result['labels']
    texture = analyze_texture(gray, settings, labels)
    texture['source_image'] = gray
    texture['segmentation_stage'] = None if result is None else result.get('stage', 'components')
    return texture
