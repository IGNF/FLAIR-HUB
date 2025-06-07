import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict, Any
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.aggregation import MeanMetric


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
            - Weighted mean IoU (train & val)
            - Per-class IoU (val)
            - Mean loss tracker (train & val)
        """
        labels = self.config['labels']
        label_configs = self.config['labels_configs']

        self.train_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(label_configs[task]['value_name']), average='weighted'
            ) for task in labels
        })

        self.val_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(label_configs[task]['value_name']), average='weighted'
            ) for task in labels
        })

        self.val_iou = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(label_configs[task]['value_name']), average=None
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

    def step(self, batch, training: bool = False):
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
        dict_logits_task, dict_logits_aux = self.forward(batch, apply_mod_dropout)

        loss_sum = 0
        all_preds, all_targets = {}, {}

        for task, logits in dict_logits_task.items():
            targets = batch[task].to(self.device)
            targets = torch.argmax(targets, dim=1) if targets.ndim == 4 else targets

            main_loss = self.criterion[task](logits, targets)
            self._check_for_invalid_loss(main_loss, task)

            main_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            aux_loss = self._compute_aux_loss(dict_logits_aux, task, targets)

            task_weight = self.config['labels_configs'][task].get('task_weight', 1.0)
            loss_sum += task_weight * (main_loss + aux_loss)

            all_preds[task] = main_preds
            all_targets[task] = targets.to(torch.int32)

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
        if task in dict_logits_aux:
            for mod in self.aux_loss_modalities:
                aux_key = f"aux_{mod}_{task}"
                if aux_key in dict_logits_aux[mod]:
                    aux_logits = dict_logits_aux[mod][task]
                    aux_loss = self.criterion[aux_key](aux_logits, targets)
                    weight = self.aux_loss_weight.get(mod, 1.0)
                    aux_losses.append(weight * aux_loss)

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
                warmup_steps = int(self.config['hyperparams']['warmup_fraction'] * self.trainer.estimated_stepping_batches)
                if self.global_step == warmup_steps:
                    print(f"\n--->---> Warmup phase completed at step {self.global_step}! LR: {scheduler.get_last_lr()[0]:.6f}")

        elif scheduler_type == "cycle_then_plateau" and not self._using_plateau:
            if self.global_step < self._warmup_scheduler.total_steps:
                self._warmup_scheduler.step()
            if self.global_step == self._warmup_scheduler.total_steps:
                self._using_plateau = True
                print(f"\n---->---> Switched to ReduceLROnPlateau at step {self.global_step}! LR: {self._warmup_scheduler.get_last_lr()[0]:.6f}")

    def on_train_epoch_end(self):
        self._log_learning_rate()
        self._log_and_reset_metrics(self.train_metrics, "train_miou")
        self.log("train_loss", self.train_loss.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.train_loss.reset()

    def _log_learning_rate(self):
        """
        Logs the current learning rate depending on the scheduler type.
        """
        scheduler_type = self._scheduler_type
        current_lr = None

        if scheduler_type == "one_cycle_lr":
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
        for task, metric in metric_dict.items():
            self.log(f"{prefix}_{task.split('-')[-1]}", metric.compute(), prog_bar=True, logger=True, sync_dist=True)
            metric.reset()

    def validation_step(self, batch, batch_idx):
        loss, all_preds, all_targets = self.step(batch, training=False)
        self.val_loss.update(loss)

        for task in all_preds:
            self.val_metrics[task].update(all_preds[task], all_targets[task])
            self.val_iou[task].update(all_preds[task], all_targets[task])
            self._log_per_class_loss(batch, task)

        return loss

    def _log_per_class_loss(self, batch, task):
        """
        Logs per-class validation loss for each task.
        Args:
            batch (dict): Input batch with labels.
            task (str): Task name.
        """
        logits = self.forward(batch)[0][task]
        targets = batch[task].to(self.device)
        targets = torch.argmax(targets, dim=1) if targets.ndim == 4 else targets

        per_class_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        class_avg_loss = torch.zeros(logits.shape[1], device=self.device)

        for class_idx in range(logits.shape[1]):
            mask = targets == class_idx
            class_avg_loss[class_idx] = per_class_loss[mask].mean() if mask.sum() > 0 else 0.0

        for class_idx, loss_val in enumerate(class_avg_loss):
            class_name = self.config['labels_configs'][task]['value_name'].get(class_idx, f"class_{class_idx}")
            self.log(f"val_loss_class_{task.split('-')[-1]}_{class_idx}_{class_name}", loss_val.item(), on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute(), prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
        total_miou = sum(self.val_metrics[task].compute().item() for task in self.val_metrics)

        for task in self.val_metrics:
            self._log_val_metrics(task)

        avg_miou = total_miou / len(self.val_metrics)
        self.log("val_miou", avg_miou, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)
        self.val_loss.reset()

        if self._scheduler_type == "cycle_then_plateau" and self._using_plateau:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self._plateau_scheduler.step(val_loss)

    def _log_val_metrics(self, task):
        """
        Logs overall and per-class IoU metrics for a given task.
        Args:
            task (str): Task name.
        """
        val_epoch_miou = self.val_metrics[task].compute()
        iou_per_class = torch.nan_to_num(self.val_iou[task].compute(), nan=0.0)

        self.log(f"val_miou_{task.split('-')[-1]}", val_epoch_miou, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True)

        class_names = self.config['labels_configs'][task]['value_name']
        for class_number, iou in enumerate(iou_per_class):
            class_name = class_names.get(class_number, f"class_{class_number}")
            self.log(f"val_iou_{task.split('-')[-1]}_{class_number}_{class_name}", iou.item(), logger=True, sync_dist=True, rank_zero_only=True)

        self.val_metrics[task].reset()
        self.val_iou[task].reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dict_logits_task, _ = self.forward(batch, apply_mod_dropout=False)
        return {
            f"preds_{task}": torch.argmax(torch.softmax(logits, dim=1), dim=1)
            for task, logits in dict_logits_task.items()
        }

    def configure_optimizers(self):
        cfg = self.config['hyperparams']
        optimizer = self._init_optimizer(cfg)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler_type = cfg.get("scheduler", None)
        warmup_fraction = cfg.get("warmup_fraction", 0.0)

        self._scheduler_type = scheduler_type

        if scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=cfg['plateau_patience'], cooldown=4, min_lr=1e-7)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}}

        if scheduler_type == "one_cycle_lr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg["learning_rate"], total_steps=total_steps,
                pct_start=warmup_fraction, cycle_momentum=False, div_factor=1000
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        if scheduler_type == "cycle_then_plateau":
            warmup_steps = int(warmup_fraction * total_steps)
            self._warmup_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg["learning_rate"], total_steps=warmup_steps,
                pct_start=1.0, cycle_momentum=False, div_factor=1000, final_div_factor=1
            )
            self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, cooldown=4, min_lr=1e-7
            )
            return {"optimizer": optimizer}

        return optimizer

    def _init_optimizer(self, cfg):
        optim_type = cfg['optimizer']
        params = self.model.parameters()
        lr = cfg["learning_rate"]

        if optim_type == 'sgd':
            return torch.optim.SGD(params, lr=lr)

        if optim_type in ['adam', 'adamw']:
            OptimClass = torch.optim.AdamW if optim_type == 'adamw' else torch.optim.Adam
            return OptimClass(
                params, lr=lr, weight_decay=cfg['optim_weight_decay'], betas=tuple(cfg['optim_betas'])
            )

        raise ValueError(f"Unsupported optimizer type: {optim_type}")
