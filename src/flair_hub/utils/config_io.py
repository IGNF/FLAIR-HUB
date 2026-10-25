import yaml
import os
import shutil

from pathlib import Path
from typing import Dict

from flair_hub.utils.distributed import rank_zero_only


RESOLVED_CONFIG_ORDER = (
    "experiment",
    "execution",
    "training",
    "hardware",
    "checkpoints",
    "logging",
    "data",
    "model",
    "segmentation_tasks",
)


class _ReadableConfigDumper(yaml.SafeDumper):
    """Render compact scalar lists while keeping nested structures readable."""


def _represent_readable_list(dumper, values):
    scalar_types = (str, int, float, bool, type(None))
    compact = len(values) <= 10 and all(
        isinstance(value, scalar_types) for value in values
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        values,
        flow_style=compact,
    )


_ReadableConfigDumper.add_representer(list, _represent_readable_list)


def format_resolved_config(public_config: dict) -> str:
    """Return a stable, human-readable and round-trippable YAML document."""
    ordered_keys = [
        key for key in RESOLVED_CONFIG_ORDER if key in public_config
    ]
    ordered_keys.extend(key for key in public_config if key not in ordered_keys)

    sections = []
    for key in ordered_keys:
        sections.append(
            yaml.dump(
                {key: public_config[key]},
                Dumper=_ReadableConfigDumper,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                width=1000,
            ).rstrip()
        )
    return "\n\n".join(sections) + "\n"


def read_config(path: str) -> Dict[str, dict]:
    """
    Reads and combines YAML configuration(s) from a file or all files in a directory.
    Args:
        path (str): Path to a single YAML file or a directory containing YAML files.
    Returns:
        Dict[str, dict]: A combined dictionary with the contents of the YAML file(s).
    """
    combined_config = {}

    def merge_file(file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration file must contain a mapping: {file_path}")
        duplicates = sorted(set(combined_config).intersection(loaded))
        if duplicates:
            raise ValueError(
                f"Duplicate top-level configuration key(s) {duplicates} in {file_path}"
            )
        combined_config.update(loaded)

    if os.path.isfile(path) and path.endswith('.yaml'):
        merge_file(path)
    elif os.path.isdir(path):
        for file_name in sorted(os.listdir(path)):
            if file_name.endswith('.yaml'):
                merge_file(os.path.join(path, file_name))
    else:
        raise ValueError(f"Invalid path: {path}. Must be a .yaml file or a directory containing .yaml files.")

    return combined_config


@rank_zero_only
def _write_resolved_config(config: dict, out_dir: Path) -> None:
    """Persist the resolved public configuration once per distributed job."""
    resolved_path = out_dir / "resolved_config.yaml"
    text = format_resolved_config(config["_resolved_config"])
    if resolved_path.exists():
        previous = resolved_path.read_text(encoding="utf-8")
        if previous == text:
            return
        import hashlib
        history = out_dir / "config_history"
        history.mkdir(exist_ok=True)
        digest = hashlib.sha256(previous.encode("utf-8")).hexdigest()[:16]
        archived = history / f"resolved_config_{digest}.yaml"
        if not archived.exists():
            archived.write_text(previous, encoding="utf-8")
    with resolved_path.open("w", encoding="utf-8") as stream:
        stream.write(text)


def prepare_output_environment(config: dict) -> Path:
    """Create the experiment directory and persist its resolved config."""
    out_dir = Path(config['paths']["out_folder"], config['paths']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(config, out_dir)
    return out_dir


@rank_zero_only
def copy_csv_and_config(config: dict, out_dir: Path) -> None:
    """
    Copy the enabled stage's CSV files to the output directory.
    Args:
        config (dict): Configuration dictionary.
        out_dir (Path): Output directory path.
    """
    csv_copy_dir = Path(out_dir, 'used_csv')
    csv_copy_dir.mkdir(parents=True, exist_ok=True)

    if config["tasks"]["train"]:
        shutil.copy(config["paths"]["train_csv"], csv_copy_dir)
        shutil.copy(config["paths"]["val_csv"], csv_copy_dir)

    if config["tasks"]["predict"]:
        shutil.copy(config["paths"]["test_csv"], csv_copy_dir)
