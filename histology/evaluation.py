"""Оцінювання збережених стадій та запис CSV/JSON."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from histology.metrics import binary_metrics, count_metrics, read_reference


def write_evaluation(directory: Path, evaluation: dict) -> None:
    """Записати метрики без NaN/Infinity: нескінченний PSNR має окрему ознаку."""
    (directory / 'evaluation.json').write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = evaluation['rows']
    if rows:
        fields = list(rows[0])
        with (directory / 'evaluation.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def evaluate_run(directory: Path, reference_path: Path, max_pixels: int) -> dict:
    """Порівняти стадії експорту; кількість ядер лише з instance-ID еталона.

    Для raw/opened масок не приписуємо істинної кількості екземплярів.
    Мітки before/after мають власні ID, тому для них рахуються N.
    """
    truth, reference_count = read_reference(reference_path, max_pixels)
    variants = []
    for stage, name in (('otsu', 'raw_mask.png'), ('opened', 'opened_mask.png')):
        path = directory / name
        if path.exists():
            with Image.open(path) as image:
                variants.append((stage, np.asarray(image).copy() > 0, None))
    for stage, name in (('before_watershed', 'before_labels.npy'), ('final', 'labels.npy')):
        path = directory / name
        if path.exists():
            labels = np.load(path, allow_pickle=False)
            variants.append((stage, labels > 0, int(np.unique(labels[labels > 0]).size)))
    if not variants and (directory / 'mask.png').exists():
        with Image.open(directory / 'mask.png') as image:
            variants.append(('mask', np.asarray(image).copy() > 0, None))
    if not variants:
        raise ValueError('Цей метод не створює маску для оцінювання. Використай all, otsu, opening або watershed.')
    rows = []
    for stage, predicted, count in variants:
        counts = count_metrics(count, reference_count) if count is not None else {
            'predicted_count': None, 'reference_count': reference_count,
            'count_absolute_error': None, 'count_relative_error': None}
        rows.append({'stage': stage, **binary_metrics(predicted, truth), **counts})
    evaluation = {'reference': str(reference_path), 'rows': rows,
                  'reference_processing': ('NPY: bool or instance IDs' if reference_path.suffix.lower() == '.npy'
                                           else 'Grayscale mask >=128; exact 0/1 masks use >0'),
                  'note': 'Raw/opened stages have pixel metrics only; SSIM is global on binary masks.'}
    write_evaluation(directory, evaluation)
    return evaluation
