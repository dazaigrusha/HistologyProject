"""Читання параметрів без зміни програмного коду."""
import json
from pathlib import Path
from histology.stains import validate_stains
from histology.segmentation import validate_segmentation


def load_config(path: Path) -> dict:
    """Прочитати JSON і перевірити параметри поточного етапу."""
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Коренем JSON має бути об’єкт.')
    for name in ('preview_width', 'preview_height', 'max_pixels'):
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name}: потрібне додатне ціле число.')
    validate_stains(config.get("stains"))
    validate_segmentation(config.get("segmentation"))
    return config
