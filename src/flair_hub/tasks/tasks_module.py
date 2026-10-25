import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

from typing import Dict, Any
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.aggregation import MeanMetric
from flair_hub.utils.class_config import model_class_indices, model_num_classes, class_weights, nomenclature_signature


def warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    warmup_ratio: float,
) -> float:
    """Return a step-based cosine multiplier with linear warm-up."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(int(step), 0), int(total_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress / total_steps))
    regular = min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if warmup_steps > 0 and progress < warmup_steps:
        warmup = warmup_ratio + (
            (1.0 - warmup_ratio) * progress / warmup_steps
        )
        return regular * warmup
    return regular


class SegmentationTask(pl.LightningModule):
    """
    PyTorch Lightning module for multi-task semantic segmentation with support for:
    - Main and auxiliary decoder outputs
    - Customizable loss functions and task weights
    - Per-class and per-task metrics (IoU, mIoU)
    - Dynamic LR scheduling including warmup and ReduceLROnPlateau
    - Predict-time softmax + argmax output per task

    Supports multiple training configurations such as modality dropout, auxiliary loss weighting,
    and various scheduler strategies (one-cycle, warmup + plateau, etc.).
    Attributes:
        model (nn.Module): Core model handling the forward logic for task and aux outputs.
        config (Dict[str, Any]): Task and training configuration dictionary.
        criterion (dict): Dictionary of loss functions for each task and auxiliary branch.
    """

    def __init__(self, model, config: Dict[str, Any], criterion=None):
        """
        Initializes the SegmentationTask module.
        Args:
            model (nn.Module): The segmentation model producing task and auxiliary outputs.
            config (Dict[str, Any]): Configuration dictionary with keys for labels, schedulers, modalities, etc.
            criterion (dict): Dictionary mapping task names (and aux keys) to their loss functions.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion

        self._setup_scheduler_flags()
        self._init_modality_flags()
        self._init_metrics()
        self._init_auxiliary_settings()

    def _setup_scheduler_flags(self):
        """
        Initializes internal flags related to learning rate schedulers.
        """
        self._scheduler_type = None
        self._using_plateau = False
        self._warmup_scheduler = None
        self._plateau_scheduler = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint['output_class_codes'] = self.output_class_codes
        checkpoint['nomenclatures'] = {
            task: nomenclature_signature(self.config['labels_configs'][task])
            for task in self.config['labels']
        }
        if self._scheduler_type == 'cycle_then_plateau':
            checkpoint['composite_scheduler'] = {
                'warmup': self._warmup_scheduler.state_dict(),
                'plateau': self._plateau_scheduler.state_dict(),
                'using_plateau': self._using_plateau,
            }
        elif self._scheduler_type == 'reduce_on_plateau':
            checkpoint['plateau_scheduler'] = self._plateau_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        signatures = checkpoint.get('nomenclatures')
        if signatures is not None:
            current = {task: nomenclature_signature(self.config['labels_configs'][task])
                       for task in self.config['labels']}
            if signatures != current:
                raise ValueError('Nomenclature changed; use initialize_from for fine-tuning')
        self._pending_scheduler_state = checkpoint.get('composite_scheduler')
        self._pending_plateau_state = checkpoint.get('plateau_scheduler')
        saved = checkpoint.get('output_class_codes')
        if saved is None:

            saved = {task: list(range(len(codes)))
                     for task, codes in self.output_class_codes.items()}
        if saved != self.output_class_codes:
            raise ValueError(
                'Checkpoint class codes differ from the current supervision. '
                'Use initialize_from for fine-tuning, not resume_from.'
            )

    def on_train_start(self):
        plateau_state = getattr(self, "_pending_plateau_state", None)
        if plateau_state and self._scheduler_type == "reduce_on_plateau":
            self._plateau_scheduler.load_state_dict(plateau_state)
        if self._scheduler_type == "one_cycle_lr":
            scheduler = self.lr_schedulers()
            plan = getattr(self, "_one_cycle_plan", None)
            if plan and scheduler.total_steps != plan["total_steps"]:
                scheduler.total_steps = plan["total_steps"]
                scheduler._schedule_phases = plan["_schedule_phases"]
                print("[SCHEDULER] One-cycle duration updated to the requested training budget.")
        state = getattr(self, '_pending_scheduler_state', None)
        if state and self._scheduler_type == 'cycle_then_plateau':
            self._warmup_scheduler.load_state_dict(state['warmup'])
            self._plateau_scheduler.load_state_dict(state['plateau'])
            self._using_plateau = state['using_plateau']
            rates = (self._plateau_scheduler.get_last_lr() if self._using_plateau
                     else self._warmup_scheduler.get_last_lr())
            for group, rate in zip(self.trainer.optimizers[0].param_groups, rates):
                group['lr'] = rate

    def _init_modality_flags(self):
        """
        Initializes whether modality dropout is active based on config.
        """
        self.mod_dropout = any(
            value > 0 for value in self.config['modalities']['modality_dropout'].values()
        )

    def _init_metrics(self):
        """
        Initializes metrics for training and validation, per task.
        Includes:
            - Macro mean IoU over non-zero-weight classes (train & val)
            - Per-class IoU (val)
            - Mean loss tracker (train & val)
        """
        labels = self.config['labels']
        label_configs = self.config['labels_configs']

        self.active_class_indices = {}
        self.ignored_class_indices = {}
        self.metric_class_names = {}
        self.output_class_codes = {}
        self.output_code_buffers = {}
        for task in labels:
            task_config = label_configs[task]
            weights = class_weights(task_config)
            model_indices = model_class_indices(task_config)
            self.output_class_codes[task] = model_indices
            buffer_name = f'_output_codes_{len(self.output_code_buffers)}'
            self.register_buffer(buffer_name, torch.tensor(model_indices, dtype=torch.long), persistent=False)
            self.output_code_buffers[task] = buffer_name
            self.active_class_indices[task] = [
                output_idx
                for output_idx, raw_idx in enumerate(model_indices)
                if weights[raw_idx] != 0
            ]
            self.metric_class_names[task] = {
                output_idx: task_config['value_name'][raw_idx]
                for output_idx, raw_idx in enumerate(model_indices)
            }
            self.ignored_class_indices[task] = [
                idx for idx, raw_idx in enumerate(model_indices)
                if weights[raw_idx] == 0
            ]

        self.train_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=max(2, model_num_classes(label_configs[task])),
                average=None,
                ignore_index=-100,
            ) for task in labels
        })

        self.val_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=max(2, model_num_classes(label_configs[task])),
                average=None,
                ignore_index=-100,
            ) for task in labels
        })

        self.val_iou = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=max(2, model_num_classes(label_configs[task])),
                average=None,
                ignore_index=-100,
            ) for task in labels
        })

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def _init_auxiliary_settings(self):
        """
        Initializes auxiliary loss modalities and weighting based on config.
        Only applies aux losses for active input modalities.
        """
        self.aux_loss_modalities = [
            mod for mod, is_active in self.config['modalities']['aux_loss'].items()
            if is_active and self.config['modalities']['inputs'].get(mod, False)
        ]
        self.aux_loss_weight = self.config['modalities']['aux_loss_weight']

    def on_train_epoch_start(self):
        self._move_metrics_to_device(self.train_metrics)
        self._move_metrics_to_device(self.val_metrics)
        self._move_metrics_to_device(self.val_iou)
        self.train_loss.to(self.device)
        self.val_loss.to(self.device)

    def _move_metrics_to_device(self, metric_dict):
        """
        Ensures metric modules are moved to the same device as the model.
        Args:
            metric_dict (nn.ModuleDict): A dictionary of torchmetrics Metric objects.
        """
        for metric in metric_dict.values():
            metric.to(self.device)

    def forward(self, batch, apply_mod_dropout: bool = False):
        """
        Runs a forward pass through the model.
        Args:
            batch (dict): Input batch including all modality inputs.
            apply_mod_dropout (bool): Whether to apply modality dropout.
        Returns:
            Tuple: (dict_logits_task, dict_logits_aux)
        """
        return self.model(batch, apply_mod_dropout)

    def _prepare_targets(self, targets: torch.Tensor, task: str) -> torch.Tensor:
        """Prepare dataset-internal indices; never interpret them as raw codes."""
        targets = targets.to(self.device)

        if targets.ndim == 4:
            if targets.shape[1] == 1:
                targets = targets[:, 0]
            else:
                valid_pixels = targets.sum(dim=1) > 0
                targets = torch.argmax(targets, dim=1)
                targets = targets.masked_fill(~valid_pixels, -100)

        if targets.ndim != 3:
            raise ValueError(
                f"Expected targets with shape [B,H,W], [B,1,H,W], or "
                f"[B,C,H,W], got {tuple(targets.shape)}"
            )

        targets = targets.long()
        ignored_indices = self.ignored_class_indices[task]
        if ignored_indices:
            targets = targets.clone()
            for class_idx in ignored_indices:
                targets[targets == class_idx] = -100

        return targets

    def _compute_active_miou(self, task: str, metric: nn.Module) -> torch.Tensor:
        """Return the macro IoU over classes whose configured weight is non-zero."""
        per_class_iou = metric.compute()
        active_indices = torch.as_tensor(
            self.active_class_indices[task],
            device=per_class_iou.device,
            dtype=torch.long,
        )
        return torch.nanmean(per_class_iou.index_select(0, active_indices))

    def step(self, batch, training: bool = False, outputs=None):
        """
        Shared logic for both training and validation steps.
        Args:
            batch (dict): Input batch with targets for all tasks.
            training (bool): Whether this is a training step (enables mod dropout).
        Returns:
            Tuple:
                - loss (torch.Tensor): Total loss.
                - all_preds (Dict[str, Tensor]): Predicted class labels per task.
                - all_targets (Dict[str, Tensor]): Ground truth labels per task.
        """
        apply_mod_dropout = self.mod_dropout if training else False
        dict_logits_task, dict_logits_aux = (
            self.forward(batch, apply_mod_dropout) if outputs is None else outputs
        )

        loss_sum = 0
        all_preds, all_targets = {}, {}

        for task, logits in dict_logits_task.items():
            targets = self._prepare_targets(batch[task], task)

            main_loss = self.criterion[task](logits, targets)
            self._check_for_invalid_loss(main_loss, task)

            main_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            aux_loss = self._compute_aux_loss(dict_logits_aux, task, targets)

            task_weight = self.config['labels_configs'][task].get('task_weight', 1.0)
            loss_sum += task_weight * (main_loss + aux_loss)

            all_preds[task] = main_preds
            all_targets[task] = targets

        return loss_sum, all_preds, all_targets

    def _compute_aux_loss(self, dict_logits_aux, task, targets):
        """
        Computes auxiliary loss for each task and active modality.
        Args:
            dict_logits_aux (dict): Dictionary of auxiliary logits.
            task (str): Task name.
            targets (torch.Tensor): Ground-truth tensor.
        Returns:
            torch.Tensor: Weighted average auxiliary loss or 0 if none.
        """
        aux_losses = []
        for mod in self.aux_loss_modalities:
            aux_key = f"aux_{mod}_{task}"
            if aux_key not in dict_logits_aux:
                continue

            aux_logits = dict_logits_aux[aux_key]
            aux_loss = self.criterion[aux_key](aux_logits, targets)
            if isinstance(self.aux_loss_weight, dict):
                weight = self.aux_loss_weight.get(mod, 1.0)
            else:
                weight = self.aux_loss_weight
            aux_losses.append(float(weight) * aux_loss)

        if aux_losses:
            stacked_loss = torch.mean(torch.stack(aux_losses))
            self._check_for_invalid_loss(stacked_loss, task, is_aux=True)
            return stacked_loss

        return torch.tensor(0.0, device=targets.device, dtype=torch.float32)

    def _check_for_invalid_loss(self, loss, task, is_aux=False):
        """
        Logs if NaN or Inf is detected in the computed loss.
        Args:
            loss (torch.Tensor): The computed loss tensor.
            task (str): Name of the task.
            is_aux (bool): Whether this is an auxiliary loss.
        """
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            kind = "auxiliary" if is_aux else "main"
            print(f"NaN or Inf detected in {kind} loss for task {task}")

    def training_step(self, batch, batch_idx):
        loss, all_preds, all_targets = self.step(batch, training=True)
        self.train_loss.update(loss)
        for task in all_preds:
            self.train_metrics[task].update(all_preds[task], all_targets[task])
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        scheduler_type = self._scheduler_type

        if scheduler_type == "one_cycle_lr":
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                warmup_steps = int(self.config['hyperparams'].get('warmup_steps', 0))
                if self.global_step == warmup_steps:
                    print(f"\n--->---> Warmup phase completed at step {self.global_step}! LR: {scheduler.get_last_lr()[0]:.6f}")

        elif scheduler_type == "cycle_then_plateau" and not self._using_plateau:
            warmup_steps = int(self.config['hyperparams']['warmup_steps'])
            if self.global_step <= warmup_steps:
                self._warmup_scheduler.step()
            if self.global_step >= warmup_steps:
                self._using_plateau = True
                print(f"\n---->---> Switched to ReduceLROnPlateau at step {self.global_step}! LR: {self._warmup_scheduler.get_last_lr()[0]:.6f}")

    def on_train_epoch_end(self):
        self._log_learning_rate()
        task_mious = self._log_and_reset_metrics(
            self.train_metrics,
            "train_miou",
        )
        train_miou = torch.stack(task_mious).mean()
        self.log(
            "train_miou",
            train_miou,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log("train_loss", self.train_loss.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.train_loss.reset()

    def _log_learning_rate(self):
        """
        Logs the current learning rate depending on the scheduler type.
        """
        scheduler_type = self._scheduler_type
        current_lr = None

        if scheduler_type in {"one_cycle_lr", "cosine"}:
            scheduler = self.lr_schedulers()
            current_lr = scheduler[0].get_last_lr()[0] if isinstance(scheduler, list) else scheduler.get_last_lr()[0]

        elif scheduler_type in ["cycle_then_plateau", "reduce_on_plateau"]:
            if self._using_plateau or scheduler_type == "reduce_on_plateau":
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            else:
                current_lr = self._warmup_scheduler.get_last_lr()[0]

        if current_lr:
            self.log("lr", current_lr, prog_bar=True, logger=True, sync_dist=True)

    def _log_and_reset_metrics(self, metric_dict, prefix):
        """
        Logs and resets task metrics after each epoch.
        Args:
            metric_dict (nn.ModuleDict): Dictionary of torchmetrics.
            prefix (str): Prefix string for log names (e.g. 'train_miou').
        """
        task_values = []
        for task, metric in metric_dict.items():
            task_value = self._compute_active_miou(task, metric)
            task_values.append(task_value)
            self.log(
                f"{prefix}_{task.split('-')[-1]}",
                task_value,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            metric.reset()
        return task_values

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss, all_preds, all_targets = self.step(batch, training=False, outputs=outputs)
        self.val_loss.update(loss)

        for task in all_preds:
            self.val_metrics[task].update(all_preds[task], all_targets[task])
            self.val_iou[task].update(all_preds[task], all_targets[task])
            self._log_per_class_loss(batch, task, outputs[0][task])

        return loss

    def _log_per_class_loss(self, batch, task, logits):
        """
        Logs per-class validation loss for each task.
        Args:
            batch (dict): Input batch with labels.
            task (str): Task name.
        """
        targets = self._prepare_targets(batch[task], task)

        per_class_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction='none',
            ignore_index=-100,
        )
        class_avg_loss = torch.zeros(logits.shape[1], device=self.device)

        for class_idx in range(logits.shape[1]):
            mask = targets == class_idx
            class_avg_loss[class_idx] = per_class_loss[mask].mean() if mask.sum() > 0 else 0.0

        for class_idx, loss_val in enumerate(class_avg_loss):
            if class_idx not in self.active_class_indices[task]:
                continue
            raw_idx = self.output_class_codes[task][class_idx]
            class_name = self.metric_class_names[task][class_idx]
            self.log(f"val_loss_class_{task.split('-')[-1]}_{raw_idx}_{class_name}", loss_val.item(), on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute(), prog_bar=True, logger=True, sync_dist=True)
        task_mious = [
            self._compute_active_miou(task, self.val_metrics[task])
            for task in self.val_metrics
        ]

        for task in self.val_metrics:
            self._log_val_metrics(task)

        avg_miou = torch.stack(task_mious).mean()
        self.log("val_miou", avg_miou, prog_bar=True, logger=True, sync_dist=True)
        self.val_loss.reset()

        if (self._scheduler_type == "reduce_on_plateau" or
                (self._scheduler_type == "cycle_then_plateau" and self._using_plateau)):
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self._plateau_scheduler.step(val_loss)

    def _log_val_metrics(self, task):
        """
        Logs overall and per-class IoU metrics for a given task.
        Args:
            task (str): Task name.
        """
        val_epoch_miou = self._compute_active_miou(task, self.val_metrics[task])
        iou_per_class = torch.nan_to_num(self.val_iou[task].compute(), nan=0.0)

        self.log(f"val_miou_{task.split('-')[-1]}", val_epoch_miou, prog_bar=True, logger=True, sync_dist=True)

        class_names = self.metric_class_names[task]
        for class_number in self.active_class_indices[task]:
            iou = iou_per_class[class_number]
            class_name = class_names.get(class_number, f"class_{class_number}")
            raw_idx = self.output_class_codes[task][class_number]
            self.log(f"val_iou_{task.split('-')[-1]}_{raw_idx}_{class_name}", iou.item(), logger=True, sync_dist=True)

        self.val_metrics[task].reset()
        self.val_iou[task].reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dict_logits_task, _ = self.forward(batch, apply_mod_dropout=False)
        return {
            f"preds_{task}": getattr(self, self.output_code_buffers[task])[
                torch.argmax(logits, dim=1)
            ]
            for task, logits in dict_logits_task.items()
        }

    def configure_optimizers(self):
        cfg = self.config['hyperparams']
        optimizer = self._init_optimizer(cfg)
        total_steps = int(cfg.get("max_steps") or self.trainer.estimated_stepping_batches)
        scheduler_type = cfg.get("scheduler", None)
        warmup_steps = int(cfg.get("warmup_steps", 0))

        self._scheduler_type = scheduler_type

        if scheduler_type == "reduce_on_plateau":
            self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=cfg['plateau_patience'], cooldown=4, min_lr=1e-7)
            return {"optimizer": optimizer}

        if scheduler_type == "one_cycle_lr":
            if not 0 < warmup_steps < total_steps:
                raise ValueError(
                    "one_cycle_lr requires warmup_steps between 1 and "
                    f"total_steps - 1; got {warmup_steps} for {total_steps} steps"
                )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg["learning_rate"], total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                cycle_momentum=False,
                div_factor=1000,
            )
            self._one_cycle_plan = scheduler.state_dict()
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        if scheduler_type == "cosine":
            schedule_steps = int(cfg.get("max_steps") or total_steps)
            min_lr_ratio = float(cfg.get("min_lr_ratio", 1e-6))
            warmup_ratio = float(cfg.get("warmup_ratio", 1e-6))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: warmup_cosine_multiplier(
                    step,
                    total_steps=schedule_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                    warmup_ratio=warmup_ratio,
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        if scheduler_type == "cycle_then_plateau":
            if not 0 < warmup_steps < total_steps:
                raise ValueError(
                    "cycle_then_plateau requires warmup_steps between 1 and "
                    f"total_steps - 1; got {warmup_steps} for {total_steps} steps"
                )
            self._warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: 0.001 + 0.999 * 0.5 * (
                    1 - math.cos(math.pi * min(step, warmup_steps) / warmup_steps))
            )
            self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5,
                patience=cfg['plateau_patience'], cooldown=4, min_lr=1e-7
            )
            return {"optimizer": optimizer}

        return optimizer

    def _init_optimizer(self, cfg):
        optim_type = cfg['optimizer']
        lr = cfg["learning_rate"]

        if optim_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)

        if optim_type in ['adam', 'adamw']:
            OptimClass = torch.optim.AdamW if optim_type == 'adamw' else torch.optim.Adam
            params = self.model.parameters()
            if optim_type == 'adamw':
                decay_params = []
                no_decay_params = []
                no_decay_keys = (
                    'absolute_pos_embed',
                    'relative_position_bias_table',
                    'norm',
                )
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if any(key in name for key in no_decay_keys):
                        no_decay_params.append(parameter)
                    else:
                        decay_params.append(parameter)
                params = [
                    {
                        'params': decay_params,
                        'weight_decay': cfg['optim_weight_decay'],
                    },
                    {'params': no_decay_params, 'weight_decay': 0.0},
                ]
            return OptimClass(
                params, lr=lr, weight_decay=cfg['optim_weight_decay'], betas=tuple(cfg['optim_betas'])
            )

        raise ValueError(f"Unsupported optimizer type: {optim_type}")
