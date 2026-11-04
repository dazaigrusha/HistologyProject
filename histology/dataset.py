"""Підготовка локальних RGB-кадрів та instance-ID масок до експериментів."""
import csv
import hashlib
import json
from pathlib import Path
import re
from tempfile import mkdtemp
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}
MASK_EXTENSIONS = {'.png', '.tif', '.tiff', '.bmp', '.npy'}


def read_instance_mask(path: Path, max_pixels: int) -> np.ndarray:
    """Прочитати ID без RGB-конвертації, щоб зберегти uint16/uint32.

    Це instance-формат: кожне додатне число — ID, 0 — фон.
    Бінарний файл не перетворюється автоматично на «еталонну кількість».
    """
    if path.suffix.lower() == '.npy':
        labels = np.load(path, allow_pickle=False, mmap_mode='r')
    else:
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise ValueError('Маска перевищує max_pixels.')
            labels = np.array(image)
    if (labels.ndim != 2 or labels.size == 0 or labels.size > max_pixels
            or labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)):
        raise ValueError('Потрібна непорожня 2D instance-маска з невід’ємними цілочисловими ID.')
    return np.asarray(labels).copy()


def _file_index(directory: Path, extensions: set, suffix: str = '') -> dict:
    """Зіставити файли за stem; неоднозначні дублікати відхиляються."""
    if not directory.is_dir():
        raise ValueError(f'Немає теки: {directory}')
    files = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem
        if suffix:
            if not key.endswith(suffix):
                continue
            key = key[:-len(suffix)]
        if key in files:
            raise ValueError(f'Неоднозначна назва {key}: {files[key]} та {path}')
        files[key] = path
    return files


def prepare_dataset(images: Path, labels: Path, directory: Path,
                    max_pixels: int = 25000000, mask_suffix: str = '',
                    group_regex: str | None = None, seed: int = 42) -> Path:
    """Зберегти точні пари й детерміноване розбиття груп 70/30.

    За замовчуванням stem — група (підходить лише для незалежних оригіналів).
    Якщо є фрагменти одного препарату/пацієнта, передай group_regex з
    однією групою захоплення для його ID. Усі фрагменти групи мають один split.
    Набір з однією групою не дозволяє незалежного holdout і відхиляється.
    """
    image_index = _file_index(images, IMAGE_EXTENSIONS)
    label_index = _file_index(labels, MASK_EXTENSIONS, mask_suffix)
    if not image_index:
        raise ValueError('Тека зображень порожня.')
    missing = sorted(set(image_index) - set(label_index))
    if missing:
        raise ValueError(f'Для кадрів немає instance-масок: {", ".join(missing[:5])}')
    try:
        pattern = None if group_regex is None else re.compile(group_regex)
    except re.error as error:
        raise ValueError(f'Некоректний group_regex: {error}') from error
    prepared = []
    for name, path in image_index.items():
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise ValueError(f'Кадр перевищує max_pixels: {name}')
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f'Спочатку узгодь орієнтацію кадру й маски: {name}')
            shape = (image.height, image.width)
        instance = read_instance_mask(label_index[name], max_pixels)
        if instance.shape != shape:
            raise ValueError(f'Розміри кадру й маски різні: {name}')
        match = None if pattern is None else pattern.search(name)
        if pattern is not None and (match is None or pattern.groups != 1):
            raise ValueError('group_regex має містити одну групу захоплення й відповідати кожній назві.')
        group = name if match is None else match.group(1)
        if not group:
            raise ValueError('Порожній ID групи.')
        # Хеш не залежить від версії генератора випадкових чисел.
        prepared.append((name, path, label_index[name], group))
    groups = sorted({item[3] for item in prepared},
                    key=lambda group: hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('Для незалежного calibration/test потрібно принаймні дві групи.')
    calibration_count = min(len(groups) - 1, max(1, round(.7 * len(groups))))
    calibration_groups = set(groups[:calibration_count])
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(mkdtemp(prefix='dataset_', dir=directory))
    (destination / 'images').mkdir()
    (destination / 'references').mkdir()
    manifests = {'all': [], 'calibration': [], 'test': []}
    provenance = []
    for name, image_path, label_path, group in prepared:
        with Image.open(image_path) as image:
            image.convert('RGB').save(destination / 'images' / f'{name}.png')
        instance = read_instance_mask(label_path, max_pixels)
        np.save(destination / 'references' / f'{name}.npy', instance, allow_pickle=False)
        row = {'image': f'images/{name}.png', 'reference': f'references/{name}.npy'}
        split = 'calibration' if group in calibration_groups else 'test'
        manifests['all'].append(row)
        manifests[split].append(row)
        # Фіксуємо лише оригінальні файли; інтерполяція/ремапінг ID не виконуються.
        provenance.append({'name': name, 'group': group, 'split': split,
            'source_image': str(image_path), 'source_mask': str(label_path),
            'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'mask_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest(),
            'nuclei': int(np.unique(instance[instance > 0]).size)})
    for split, rows in manifests.items():
        with (destination / f'{split}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image', 'reference'])
            writer.writeheader()
            writer.writerows(rows)
    data = {'seed': seed, 'images': len(prepared), 'groups': len(groups),
            'calibration_images': len(manifests['calibration']), 'test_images': len(manifests['test']),
            'group_regex': group_regex, 'items': provenance,
            'note': 'Split units must represent independent specimens/patients. No evaluation was run by import.'}
    (destination / 'dataset.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination
