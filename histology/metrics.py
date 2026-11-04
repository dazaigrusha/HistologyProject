"""Ручні метрики бінарних масок і похибка кількості об’єктів."""
import math
from pathlib import Path
import numpy as np
from histology.image_io import load_image


def binary_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """Dice/IoU та похибки для двох bool-масок однакового розміру.

    Порожня пара: Dice=IoU=1. Невизначені precision/recall — None.
    MSE та PSNR обчислюються на 0/1, а не між RGB-оригіналом і маскою.
    SSIM — явно глобальна формула з population-дисперсією, C1=.01²,
    C2=.03²; це не стандартне усереднення локальних гаусових вікон.
    """
    if (predicted.dtype != np.bool_ or reference.dtype != np.bool_
            or predicted.ndim != 2 or predicted.shape != reference.shape or predicted.size == 0):
        raise ValueError('Метрики очікують дві непорожні bool-маски однакового розміру.')
    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    tn = int(predicted.size - tp - fp - fn)
    p, r = tp + fp, tp + fn
    union = tp + fp + fn
    mse = (fp + fn) / predicted.size
    mean_p, mean_r = p / predicted.size, r / predicted.size
    variance_p, variance_r = mean_p * (1 - mean_p), mean_r * (1 - mean_r)
    covariance = tp / predicted.size - mean_p * mean_r
    c1, c2 = .01 ** 2, .03 ** 2
    ssim = ((2 * mean_p * mean_r + c1) * (2 * covariance + c2)
            / ((mean_p ** 2 + mean_r ** 2 + c1) * (variance_p + variance_r + c2)))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'dice': 1.0 if p + r == 0 else 2 * tp / (p + r),
            'iou': 1.0 if union == 0 else tp / union,
            'precision': None if p == 0 else tp / p,
            'recall': None if r == 0 else tp / r,
            'mse_binary': mse, 'psnr_db': None if mse == 0 else -10 * math.log10(mse),
            'psnr_infinite': mse == 0, 'ssim_global': float(ssim),
            'reference_empty': r == 0, 'predicted_empty': p == 0}


def count_metrics(predicted: int, reference: int | None) -> dict:
    """Похибка N без зіставлення екземплярів; Nref=0 не має відносної похибки."""
    return {'predicted_count': predicted, 'reference_count': reference,
            'count_absolute_error': None if reference is None else abs(predicted - reference),
            'count_relative_error': None if reference in (None, 0) else abs(predicted - reference) / reference}


def read_reference(path: Path, max_pixels: int) -> tuple:
    """PNG/BMP/TIFF: строго бінарна маска; NPY: bool-маска або instance-ID.

    У цілочисловому NPY кожне додатне значення — один розмічений об’єкт,
    нуль — фон. PNG не дає істинної кількості ядер при їхньому дотику.
    Масштаб не змінюється; для зображень враховується EXIF, як при читанні оригіналу.
    """
    if path.suffix.lower() == '.npy':
        labels = np.load(path, allow_pickle=False, mmap_mode='r')
        if labels.ndim != 2 or labels.size == 0 or labels.size > max_pixels:
            raise ValueError('Еталон NPY має бути непорожнім 2D-масивом у межах max_pixels.')
        if labels.dtype == np.bool_:
            return np.asarray(labels).copy(), None
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError('Еталон NPY: bool або невід’ємні цілочислові ID.')
        count = int(np.unique(labels[labels > 0]).size)
        return np.asarray(labels > 0), count
    rgb = np.asarray(load_image(path, max_pixels))
    values = np.unique(rgb)
    if not np.all(rgb == rgb[..., :1]) or not (np.all(np.isin(values, [0, 1]))
                                              or np.all(np.isin(values, [0, 255]))):
        raise ValueError('Еталон-зображення має бути бінарним 0/1 або 0/255 без JPEG-артефактів.')
    return rgb[..., 0] > 0, None


def evaluate_result(result: dict, reference_path: Path, max_pixels: int) -> dict:
    """Порівняти поточні мітки й, за наявності, мітки до Watershed."""
    truth, reference_count = read_reference(reference_path, max_pixels)
    variants = {'current': result['labels']}
    if 'before_labels' in result:
        variants = {'before_watershed': result['before_labels'], 'after_watershed': result['labels']}
    rows = []
    for stage, labels in variants.items():
        count = int(np.unique(labels[labels > 0]).size)
        rows.append({'stage': stage, **binary_metrics(labels > 0, truth),
                     **count_metrics(count, reference_count)})
    return {'reference': str(reference_path), 'rows': rows,
            'note': 'Pixel overlap and count error; no instance matching or instance F1.'}
