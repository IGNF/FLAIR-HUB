"""Finalize a resumable checkpoint before removing temporary recovery files."""

import os
import shutil
import warnings
from pathlib import Path


def auto_resume_path(recovery_path, out_dir):
    if recovery_path and Path(recovery_path).is_file():
        return str(recovery_path)
    return str(Path(out_dir) / 'checkpoints' / 'last.ckpt')


def finalize_recovery(trainer, config, out_dir):
    recovery = config['saving'].get('recovery_checkpoint', {})
    status = getattr(trainer.state.status, 'value', trainer.state.status)
    if status != 'finished' or not recovery.get('enabled', False):
        return
    if (not config['saving']['ckpt_save_also_last']
            or config['saving']['ckpt_weights_only']):
        return

    root = Path(out_dir).resolve()
    recovery_path = root / recovery.get('dir_name', 'recovery_checkpoints')
    resolved_recovery = recovery_path.resolve()
    checkpoint_dir = (root / 'checkpoints').resolve()
    if (recovery_path.is_symlink() or resolved_recovery == root
            or root not in resolved_recovery.parents
            or resolved_recovery == checkpoint_dir
            or resolved_recovery in checkpoint_dir.parents
            or checkpoint_dir in resolved_recovery.parents):
        if trainer.is_global_zero:
            warnings.warn('Unsafe recovery directory; keeping recovery files.')
        return

    final_path = checkpoint_dir / 'last.ckpt'
    temporary_path = checkpoint_dir / 'last-finalizing.ckpt'
    trainer.save_checkpoint(str(temporary_path), weights_only=False)
    trainer.strategy.barrier('final_checkpoint_saved')
    if trainer.is_global_zero:
        try:
            import torch
            checkpoint = torch.load(
                temporary_path, map_location='cpu', mmap=True, weights_only=False
            )
            valid = (
                checkpoint.get('global_step') == trainer.global_step
                and bool(checkpoint.get('state_dict'))
                and bool(checkpoint.get('optimizer_states'))
                and 'lr_schedulers' in checkpoint
                and 'loops' in checkpoint
            )
            del checkpoint
            if not valid:
                raise ValueError('Final checkpoint is incomplete or has the wrong step')
            os.replace(temporary_path, final_path)
            if recovery_path.exists():
                shutil.rmtree(resolved_recovery)
                print(f'Removed recovery checkpoints after final save: {resolved_recovery}')
        except Exception as exc:
            warnings.warn(f'Final checkpoint cleanup failed; recovery retained: {exc}')
    trainer.strategy.barrier('recovery_cleanup_finished')
