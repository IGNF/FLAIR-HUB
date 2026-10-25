import argparse
import sys
import time
from pathlib import Path

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "--conf_folder",
    help="Path to a YAML config file or a directory containing YAML config files",
    required=True,
)

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import flair_hub

def main():
    """Run the configured training and prediction workflow."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = argParser.parse_args()
    startup_started = time.perf_counter()

    from flair_hub.utils.config_io import (
        copy_csv_and_config,
        prepare_output_environment,
        read_config,
    )
    from flair_hub.utils.config_schema import load_and_compile_public_config
    from flair_hub.utils.distributed import is_global_zero
    from flair_hub.utils.messaging import (
        BootstrapStream,
        DiagnosticLogger,
        Logger,
        end_msg,
        start_msg,
        status_step,
    )

    terminal = sys.stdout
    terminal_stderr = sys.stderr
    bootstrap = BootstrapStream(terminal)
    stderr_bootstrap = BootstrapStream(terminal_stderr)
    if is_global_zero():
        sys.stdout = bootstrap

    sys.stderr = stderr_bootstrap

    start_msg()

    with status_step("Reading configuration files"):
        public_config = read_config(args.conf_folder)
    if is_global_zero():
        config_path = Path(args.conf_folder).expanduser().resolve()
        print(f"[INFO] Config path: {config_path}", flush=True)

    with status_step("Validating and compiling configuration"):
        config = load_and_compile_public_config(public_config)

    with status_step("Preparing output directory and resolved_config.yaml"):
        out_dir = prepare_output_environment(config)

    run_logger = None
    diagnostic_logger = None
    with status_step("Initializing run loggers"):
        if is_global_zero():
            run_logger = Logger(
                Path(
                    out_dir,
                    f'flair-compute_{config["paths"]["out_model_name"]}.log',
                ).as_posix(),
                terminal=terminal,
            )
        diagnostic_logger = DiagnosticLogger(
            Path(
                out_dir,
                f'flair-compute_{config["paths"]["out_model_name"]}'
                '_err-warn.log',
            ).as_posix(),
            terminal=terminal_stderr,
        )
        diagnostic_logger.write_log_only(stderr_bootstrap.getvalue())
        sys.stderr = diagnostic_logger

    if is_global_zero():
        run_logger.write_log_only(bootstrap.getvalue())
        sys.stdout = run_logger
        print(f"[INFO] Run log: {run_logger.filename}", flush=True)
        print(
            f"[INFO] Error/warning log: {diagnostic_logger.filename}",
            flush=True,
        )
        print(f"[INFO] Package source: {flair_hub.__file__}", flush=True)
        print(f"[INFO] Package variant: {flair_hub.VARIANT}", flush=True)

    with status_step("Importing PyTorch, Lightning, model and data modules"):
        from flair_hub.tasks.stages import training_stage, predict_stage
        from flair_hub.tasks.module_setup import build_data_module
        from flair_hub.data.utils_data.paths import get_datasets
        from flair_hub.utils.config_display import print_recap

    with status_step("Reading and indexing configured dataset splits"):
        dict_train, dict_val, dict_test = get_datasets(config)

    if config['saving']["cp_csv_and_conf_to_output"]:
        with status_step("Archiving configured input CSV files"):
            copy_csv_and_config(config, out_dir)

    with status_step("Building the Lightning data module"):
        dm = build_data_module(
            config,
            dict_train=dict_train,
            dict_val=dict_val,
            dict_test=dict_test,
        )

    if is_global_zero():
        print(
            f"[INFO] Startup completed in "
            f"{time.perf_counter() - startup_started:.2f}s.",
            flush=True,
        )
        print(flush=True)

    print_recap(config, dict_train, dict_val, dict_test)

    trained_state_dict = None


    if config['tasks']['train']:
        trained_state_dict = training_stage(config, dm, out_dir)


    if config['tasks'].get('predict') or config['tasks'].get('metrics_only'):
        out_dir_predict = Path(out_dir, 'results_' + config['paths']["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)
    else:
        print("[WARNING] Neither prediction nor metrics_only was enabled. Finishing.")

    end_msg()

if __name__ == "__main__":
    main()
