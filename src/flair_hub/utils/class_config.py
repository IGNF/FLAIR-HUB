"""Source codes are public; consecutive model indices are strictly internal."""
import numpy as np

IGNORE_INDEX = -100


def class_codes(task_config):
    return sorted(int(code) for code in task_config['value_name'])


def class_weights(task_config):
    """Weights keyed by source code, never by position in a list."""
    config = task_config.get('value_weights', {})
    exceptions = config.get('default_exceptions', {}) or {}
    return {code: float(exceptions.get(code, config.get('default', 1.0)))
            for code in class_codes(task_config)}


def active_class_indices(task_config):
    return [code for code, weight in class_weights(task_config).items()
            if weight > 0]


def model_class_indices(task_config):
    """Position is the model index; value is the source code."""
    codes = active_class_indices(task_config)
    if not codes:
        raise ValueError('At least one active class is required')
    return codes


def label_lookup(task_config):
    return {code: index for index, code in enumerate(model_class_indices(task_config))}


def encode_labels(labels, lookup):
    """Sparse vectorized mapping; memory depends on class count, not max code."""
    codes = np.asarray(sorted(lookup), dtype=np.int64)
    values = np.asarray([lookup[int(code)] for code in codes], dtype=np.int64)
    indices = np.searchsorted(codes, labels)
    bounded = np.minimum(indices, len(codes) - 1)
    valid = (indices < len(codes)) & (codes[bounded] == labels)
    return np.where(valid, values[bounded], IGNORE_INDEX).astype(np.int64)


def model_num_classes(task_config):
    return len(model_class_indices(task_config))


def output_dtype(task_config):
    maximum = max(class_codes(task_config))
    return 'uint8' if maximum <= 255 else 'uint16' if maximum <= 65535 else 'uint32'


def nomenclature_signature(task_config):
    return {'version': task_config.get('nomenclature_version'),
            'classes': [[code, str(task_config['value_name'][code])]
                        for code in model_class_indices(task_config)]}
