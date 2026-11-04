"""Експорт каналів, маркерів, Watershed-міток і вимірювань."""
import csv
import json
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
from PIL import Image
from histology.analysis import distance_image, label_image, marker_image
from histology.measurements import measure_regions, summarize_regions
from histology.stains import transmittance_image
from histology.texture_export import write_texture
from histology.evaluation import write_evaluation

OBJECT_FIELDS = ['id', 'area_px', 'x_min', 'y_min', 'x_max', 'y_max',
                 'centroid_x', 'centroid_y', 'touches_border', 'perimeter_px',
                 'circularity', 'equivalent_diameter_px', 'area_um2', 'perimeter_um',
                 'equivalent_diameter_um']


def write_objects(path: Path, objects: list) -> None:
    """Зберегти UTF-8 BOM CSV; порожній список усе одно має заголовок."""
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=OBJECT_FIELDS)
        writer.writeheader()
        writer.writerows(objects)


def save_results(directory: Path, source_name: str, size: tuple, maps: tuple,
                 config: dict, result: dict | None = None,
                 texture: dict | None = None) -> Path:
    """Зберегти дані поточної стадії в нову теку, не перезаписуючи минулі запуски.

    Після помилки запису можливі часткові файли у створеній теці.
    Файли маркерів/Watershed створюються лише після третьої кнопки.
    """
    destination = Path(mkdtemp(prefix='histology_', dir=directory))
    for name, channel in zip(('hematoxylin', 'eosin'), maps):
        Image.fromarray(transmittance_image(channel)).save(destination / f'{name}.png')
        np.save(destination / f'{name}_od.npy', channel, allow_pickle=False)
    metadata = {'source': source_name, 'size': list(size),
                'method': 'two_stain_od_least_squares_clipped', 'stains': config['stains'],
                'measurements': config['measurements'],
                'png_channels': 'uint8 transmittance: round(255*exp(-concentration))',
                'npy_channels': 'float32 concentration, original resolution'}
    if result is not None:
        Image.fromarray(result['strength']).save(destination / 'nuclear_strength.png')
        for key in ('raw_mask', 'opened_mask', 'mask'):
            Image.fromarray(result[key].astype(np.uint8) * 255).save(destination / f'{key}.png')
        result['overlay'].save(destination / 'overlay.png')
        np.save(destination / 'labels.npy', result['labels'], allow_pickle=False)
        label_image(result['labels']).save(destination / 'labels.png')
        scale = config['measurements']['pixel_size_um']
        objects = measure_regions(result['labels'], scale)
        write_objects(destination / 'objects.csv', objects)
        summary = {'after': summarize_regions(objects, result['labels'].shape, scale)}
        metadata.update({'stage': result.get('stage', 'components'),
                         'segmentation': config['segmentation'],
                         'otsu_threshold': result['otsu_threshold'],
                         'effective_threshold': result['effective_threshold'],
                         'candidate_count': result['count'],
                         'area_unit': 'pixel^2', 'perimeter_method': 'exposed_pixel_edges_4',
                         'bbox_max': 'exclusive', 'density_reference': 'whole_image_frame',
                         'note': 'Candidate nuclei; accuracy on real histological masks not yet evaluated.'})
        if result.get('stage') == 'watershed':
            for key in ('distance', 'markers', 'watershed_labels', 'before_labels'):
                np.save(destination / f'{key}.npy', result[key], allow_pickle=False)
            distance_image(result['distance']).save(destination / 'distance.png')
            marker_image(result['distance'], result['markers']).save(destination / 'markers.png')
            result['before_overlay'].save(destination / 'before_overlay.png')
            write_objects(destination / 'objects_before.csv', result['before_objects'])
            summary['before'] = result['before_summary']
            summary['markers'] = result['marker_count']
            summary['basins_before_filter'] = result['basin_count_before_filter']
            metadata['watershed'] = config['watershed']
            metadata['before_candidate_count'] = result['before_count']
            metadata['watershed_partition'] = 'no zero-valued lines inside input foreground'
        (destination / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if result is not None and result.get('evaluation') is not None:
        write_evaluation(destination, result['evaluation'])
    if texture is not None:
        write_texture(destination, texture)
        metadata['texture'] = texture['settings']
    (destination / 'parameters.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return destination
