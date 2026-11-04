"""Окремий запуск п’яти методів або повного сценарію без графічного вікна."""
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
import numpy as np
from PIL import Image
from histology.analysis import distance_image, label_image, marker_image
from histology.export import save_results
from histology.image_io import load_image
from histology.measurements import filter_regions, summarize_regions
from histology.segmentation import binary_opening, otsu_threshold
from histology.stains import separate_stains
from histology.texture import rgb_to_gray
from histology.texture_export import write_texture
from histology.watershed import split_touching
from histology.workflow import analyze_image, split_image, texture_image

METHODS = ('stains', 'otsu', 'opening', 'watershed', 'lbp', 'all')


def _binary_input(rgb: np.ndarray) -> np.ndarray:
    """Бінарний вхід має містити лише чорні та білі RGB-пікселі."""
    if not np.all((rgb == 0) | (rgb == 255)) or not np.all(rgb == rgb[..., :1]):
        raise ValueError('Для opening/watershed потрібна чорно-біла маска 0/255, не кольоровий оригінал.')
    return rgb[..., 0] == 255


def run_method(source: Path, method: str, directory: Path, config: dict) -> Path:
    """Запустити один метод; зберегти вхідні параметри й час операції в run.json.

    Час охоплює читання, обчислення та запис результатів, крім самого run.json.
    Це час запуску, а не метрика якості сегментації.
    """
    if method not in METHODS:
        raise ValueError(f'Невідомий метод: {method}')
    started = perf_counter()
    image = load_image(source, config['max_pixels'])
    rgb = np.asarray(image)
    directory.mkdir(parents=True, exist_ok=True)
    destination = None
    prerequisites = []
    if method in ('stains', 'all'):
        maps = separate_stains(rgb, config['stains'])
        result = texture = None
        if method == 'all':
            result = analyze_image(rgb, maps[0], config['segmentation'], config['measurements'])
            result = split_image(rgb, result, config)
            texture = texture_image(rgb, maps, result, config['texture'])
            texture['source_name'] = source.name
        destination = save_results(directory, source.name, image.size, maps, config, result, texture)
    elif method == 'lbp':
        if config['texture']['scope'] == 'objects':
            raise ValueError('Окремий lbp використовує scope=whole. Для об’єктів запусти --method all.')
        maps = None
        if config['texture']['source'] == 'hematoxylin':
            maps = separate_stains(rgb, config['stains'])
            prerequisites.append('H&E separation for hematoxylin source')
        texture = texture_image(rgb, maps, None, config['texture'])
        texture['source_name'] = source.name
        destination = Path(mkdtemp(prefix='lbp_', dir=directory))
        write_texture(destination, texture)
    elif method == 'otsu':
        gray = rgb_to_gray(rgb)
        threshold = otsu_threshold(gray)
        # Самостійний Оцу: світлий клас > t; нижню межу з H-сценарію не додаємо.
        mask = gray > threshold
        destination = Path(mkdtemp(prefix='otsu_', dir=directory))
        Image.fromarray(gray).save(destination / 'source_gray.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(destination / 'mask.png')
        (destination / 'threshold.json').write_text(json.dumps(
            {'threshold': threshold, 'foreground': 'gray>threshold'}), encoding='utf-8')
    else:
        mask = _binary_input(rgb)
        if method == 'opening':
            opened = binary_opening(mask, config['segmentation']['opening_radius'])
            destination = Path(mkdtemp(prefix='opening_', dir=directory))
            Image.fromarray(opened.astype(np.uint8) * 255).save(destination / 'opened_mask.png')
        else:
            result = split_touching(mask, config['watershed'], config['segmentation']['connectivity'])
            labels, objects = filter_regions(result['watershed_labels'], config['segmentation'],
                                              config['measurements']['pixel_size_um'])
            destination = Path(mkdtemp(prefix='watershed_', dir=directory))
            for name in ('distance', 'markers', 'watershed_labels'):
                np.save(destination / f'{name}.npy', result[name], allow_pickle=False)
            np.save(destination / 'labels.npy', labels, allow_pickle=False)
            label_image(labels).save(destination / 'labels.png')
            distance_image(result['distance']).save(destination / 'distance.png')
            marker_image(result['distance'], result['markers']).save(destination / 'markers.png')
            from histology.export import write_objects
            write_objects(destination / 'objects.csv', objects)
            summary = summarize_regions(objects, labels.shape, config['measurements']['pixel_size_um'])
            (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        Image.fromarray(mask.astype(np.uint8) * 255).save(destination / 'input_mask.png')
    metadata = {'input': source.name, 'method': method, 'config': config,
                'automatic_prerequisites': prerequisites, 'size': list(image.size),
                'elapsed_seconds_including_io': perf_counter() - started}
    (destination / 'run.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return destination
