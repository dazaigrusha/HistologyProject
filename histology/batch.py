"""Пакетний запуск за CSV-маніфестом з ізольованими помилками файлів."""
import csv
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from histology.cli import run_method
from histology.evaluation import evaluate_run


def run_batch(manifest: Path, directory: Path, config: dict,
              method: str = 'all') -> Path:
    """Обробити image,reference з CSV; відносні шляхи — від теки CSV.

    Відсутній еталон не підмінюється синтетикою: метрики лишаються порожніми.
    Помилка окремого кадру не перериває всю вибірку. Запуски послідовні.
    """
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'image' not in reader.fieldnames:
            raise ValueError('CSV вибірки має містити стовпець image; reference необов’язковий.')
        records = list(reader)
    if not records:
        raise ValueError('CSV вибірки порожній.')
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(mkdtemp(prefix='batch_', dir=directory))
    items = destination / 'items'; items.mkdir()
    (destination / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
    statuses, metrics = [], []
    for index, record in enumerate(records, 1):
        started = perf_counter()
        name = (record.get('image') or '').strip()
        reference_name = (record.get('reference') or '').strip()
        status = {'row': index, 'image': name, 'reference': reference_name,
                  'status': '', 'output': '', 'error': '', 'elapsed_seconds': 0.0}
        try:
            if not name:
                raise ValueError('Порожній шлях image.')
            source = Path(name)
            if not source.is_absolute():
                source = manifest.parent / source
            folder = run_method(source, method, items, config)
            status['output'] = str(folder.relative_to(destination))
            status['status'] = 'ok_no_reference'
            if reference_name:
                status['status'] = 'evaluation_error'
                reference = Path(reference_name)
                if not reference.is_absolute():
                    reference = manifest.parent / reference
                evaluation = evaluate_run(folder, reference, config['max_pixels'])
                metrics.extend({'row': index, 'image': name, **row} for row in evaluation['rows'])
                status['status'] = 'ok_evaluated'
        except Exception as error:  # Ізолюємо несправний файл, фіксуючи помилку в журналі.
            if not status['status']:
                status['status'] = 'processing_error'
            status['error'] = f'{type(error).__name__}: {error}'
        status['elapsed_seconds'] = perf_counter() - started
        statuses.append(status)
    _write_csv(destination / 'batch_status.csv', statuses)
    if metrics:
        _write_csv(destination / 'metrics.csv', metrics)
    stages = sorted({item['stage'] for item in metrics})
    groups = []
    metric_names = ('dice', 'iou', 'precision', 'recall', 'mse_binary', 'psnr_db',
                    'ssim_global', 'count_absolute_error', 'count_relative_error')
    for stage in stages:
        rows = [item for item in metrics if item['stage'] == stage]
        group = {'stage': stage, 'evaluated_images': len(rows),
                 'empty_references': sum(item['reference_empty'] for item in rows),
                 'perfect_psnr_cases': sum(item['psnr_infinite'] for item in rows)}
        for name in metric_names:
            values = [item[name] for item in rows if item[name] is not None]
            group[f'mean_{name}'] = None if not values else sum(values) / len(values)
            group[f'n_{name}'] = len(values)
        nonempty = [item['dice'] for item in rows if not item['reference_empty']]
        group['mean_dice_nonempty_reference'] = None if not nonempty else sum(nonempty) / len(nonempty)
        groups.append(group)
    if groups:
        _write_csv(destination / 'comparison.csv', groups)
    paired = []
    for index in range(1, len(records) + 1):
        by_stage = {item['stage']: item for item in metrics if item['row'] == index}
        if 'before_watershed' in by_stage and 'final' in by_stage:
            before, after = by_stage['before_watershed'], by_stage['final']
            delta_count = None if before['count_absolute_error'] is None else (
                after['count_absolute_error'] - before['count_absolute_error'])
            paired.append({'row': index, 'image': before['image'],
                           'dice_delta_after_minus_before': after['dice'] - before['dice'],
                           'count_absolute_error_delta': delta_count})
    if paired:
        _write_csv(destination / 'paired_comparison.csv', paired)
    summary = {'method': method, 'rows': len(statuses),
               'processed': sum(item['status'] != 'processing_error' for item in statuses),
               'evaluated': sum(item['status'] == 'ok_evaluated' for item in statuses),
               'processing_errors': sum(item['status'] == 'processing_error' for item in statuses),
               'evaluation_errors': sum(item['status'] == 'evaluation_error' for item in statuses),
               'stages': groups, 'aggregation': 'macro mean over available evaluated images',
               'note': 'No evaluation means no quality claim. Empty-reference Dice=1 for empty prediction.'}
    (destination / 'batch_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return destination


def _write_csv(path: Path, rows: list) -> None:
    """Зберегти непорожню таблицю з узгодженим порядком полів."""
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
