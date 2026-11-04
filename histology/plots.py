"""Наукові графіки з реальних таблиць пакетного звіту, без нових обчислень CV."""
import csv
import math
from pathlib import Path


def plot_comparison(directory: Path) -> Path:
    """Побудувати macro Dice/IoU та MAE кількості з comparison.csv.

    Потрібна необов’язкова залежність matplotlib. Порожні/невизначені
    оцінки не підмінюються нулями. Значення n беруться зі звіту.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with (directory / 'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('comparison.csv порожній: спочатку виконай batch із еталонами.')

    def number(row, key):
        raw = row.get(key, '')
        if raw == '':
            return None
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f'Некоректне значення {key}.')
        return value

    figure = Figure(figsize=(11, 4.8), layout='constrained')
    FigureCanvasAgg(figure)
    overlap, count_axis = figure.subplots(1, 2)
    stages = [row['stage'] for row in rows]
    positions = list(range(len(rows)))
    for offset, name, color in ((-.18, 'dice', '#6752a3'), (.18, 'iou', '#39948b')):
        for index, row in enumerate(rows):
            value = number(row, f'mean_{name}')
            if value is None:
                continue
            overlap.bar(index + offset, value, width=.34, color=color,
                        label=name.upper() if index == 0 else None)
    overlap.set_xticks(positions, stages, rotation=25, ha='right')
    overlap.set_ylim(0, 1.05)
    overlap.set_ylabel('Macro mean (0-1)')
    overlap.set_title('Pixel overlap; empty references included')
    overlap.legend()
    overlap.grid(axis='y', alpha=.2)
    available = [(row, number(row, 'mean_count_absolute_error')) for row in rows]
    available = [(row, value) for row, value in available if value is not None]
    if available:
        x = list(range(len(available)))
        values = [value for _, value in available]
        count_axis.bar(x, values, color='#b87d42')
        count_axis.set_xticks(x, [row['stage'] for row, _ in available], rotation=25, ha='right')
        for index, (row, value) in enumerate(available):
            count_axis.annotate(f"n={row['n_count_absolute_error']}", (index, value),
                                xytext=(0, 5), textcoords='offset points', ha='center')
        count_axis.set_ylabel('Mean absolute count error')
        count_axis.set_ylim(0, max(1, max(values) * 1.25))
    else:
        count_axis.text(.5, .5, 'No instance-ID references\nCount error unavailable', ha='center', va='center')
        count_axis.set_axis_off()
    count_axis.set_title('Nucleus count; lower is better')
    figure.suptitle('Batch comparison; inspect n and per-image metrics before interpreting')
    path = directory / 'comparison.png'
    figure.savefig(path, dpi=180)
    return path
