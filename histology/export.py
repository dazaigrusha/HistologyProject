"""Збереження каналів, масок, міток і таблиці кандидатів."""
import csv
import json
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
from PIL import Image
from histology.stains import transmittance_image

OBJECT_FIELDS = ['id', 'area_px', 'x_min', 'y_min', 'x_max', 'y_max',
                 'centroid_x', 'centroid_y', 'touches_border']


def save_results(directory: Path, source_name: str, size: tuple, maps: tuple,
                 config: dict, result: dict | None = None) -> Path:
    """Створити унікальну теку; повернути її шлях після успішного запису.

    Маски: біле=об’єкт, чорне=фон. labels.npy: int32, 0=фон, ID=рядок CSV.
    При помилці запису можуть залишитися часткові файли в новій теці.
    """
    destination = Path(mkdtemp(prefix='histology_', dir=directory))
    for name, channel in zip(('hematoxylin', 'eosin'), maps):
        Image.fromarray(transmittance_image(channel)).save(destination / f'{name}.png')
        np.save(destination / f'{name}_od.npy', channel, allow_pickle=False)
    metadata = {'source': source_name, 'size': list(size),
                'method': 'two_stain_od_least_squares_clipped', 'stains': config['stains'],
                'png_channels': 'uint8 transmittance: round(255*exp(-concentration))',
                'npy_channels': 'float32 concentration, original resolution'}
    if result is not None:
        Image.fromarray(result['strength']).save(destination / 'nuclear_strength.png')
        for key in ('raw_mask', 'opened_mask', 'mask'):
            Image.fromarray(result[key].astype(np.uint8) * 255).save(destination / f'{key}.png')
        result['overlay'].save(destination / 'overlay.png')
        np.save(destination / 'labels.npy', result['labels'], allow_pickle=False)
        with (destination / 'objects.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=OBJECT_FIELDS)
            writer.writeheader()
            writer.writerows(result['objects'])
        metadata.update({'segmentation': config['segmentation'],
                         'otsu_threshold': result['otsu_threshold'],
                         'effective_threshold': result['effective_threshold'],
                         'candidate_count': result['count'],
                         'area_unit': 'pixel^2', 'bbox_max': 'exclusive',
                         'note': 'Connected components, not validated nucleus count; touching nuclei may merge.'})
    (destination / 'parameters.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination
