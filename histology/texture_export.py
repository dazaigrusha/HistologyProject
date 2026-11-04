"""Запис карти, допустимих центрів, гістограми й ознак LBP."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from histology.texture import histogram_image


def write_texture(destination: Path, texture: dict) -> None:
    """Зберегти LBP у готову теку; ID об’єктів відповідають поточним labels.npy."""
    destination.mkdir(parents=True, exist_ok=True)
    for key, filename in (('codes', 'lbp_codes.png'), ('source_image', 'lbp_source.png')):
        Image.fromarray(texture[key]).save(destination / filename)
    Image.fromarray(texture['valid_mask'].astype(np.uint8) * 255).save(destination / 'lbp_valid_mask.png')
    np.save(destination / 'lbp_codes.npy', texture['codes'], allow_pickle=False)
    histogram_image(texture['histogram_counts']).save(destination / 'lbp_histogram.png')
    with (destination / 'lbp_histogram.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['code', 'count', 'probability'])
        for code, (count, probability) in enumerate(zip(texture['histogram_counts'], texture['histogram_probabilities'])):
            writer.writerow([code, int(count), float(probability)])
    data = {'source': texture.get('source_name'), 'settings': texture['settings'], 'features': texture['features'],
            'segmentation_stage': texture['segmentation_stage'],
            'border': 'exclude outer 1-pixel frame', 'comparison': 'neighbor>=center',
            'bit_order': 'NW,N,NE,E,SE,S,SW,W (bits 0..7)',
            'histogram_counts': texture['histogram_counts'].tolist(), 'objects': texture['objects']}
    (destination / 'lbp_features.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (destination / 'lbp_objects.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['id', 'valid_centers', 'entropy_bits', 'energy', 'dominant_code']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(texture['objects'])
