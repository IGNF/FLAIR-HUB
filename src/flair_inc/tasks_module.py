import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.aggregation import MeanMetric
from typing import Dict, Any


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        config: Dict[str, Any],
        criterion=None,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()

        self.model = model
        self.config = config
        self.criterion = criterion  
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.mod_dropout = any(value > 0 for value in config['modalities']['modality_dropout'].values())
       
        self.train_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(config['labels_configs'][task]['value_name']), 
                average='weighted'
            )
            for task in config['labels']
        })
        self.val_metrics = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(config['labels_configs'][task]['value_name']), 
                average='weighted'
            )
            for task in config['labels']
        })
        self.val_iou = nn.ModuleDict({
            task: MulticlassJaccardIndex(
                num_classes=len(config['labels_configs'][task]['value_name']), 
                average=None
            )
            for task in config['labels']
        })

        self.train_loss = MeanMetric().to(self.device)  
        self.val_loss = MeanMetric().to(self.device)

        # Get auxiliary modalities that are both in `aux_loss` and `input`
        self.aux_loss_modalities = [
            mod for mod, is_active in config['modalities']['aux_loss'].items()
            if is_active and config['modalities']['inputs'].get(mod, False)
        ]
        self.aux_loss_weight = config['modalities']['aux_loss_weight']


    def on_train_epoch_start(self):
        """Ensure all metrics are on the correct device before the epoch starts."""
        for metric in self.train_metrics.values():
            metric.to(self.device)
        for metric in self.val_metrics.values():
            metric.to(self.device)
        for metric in self.val_iou.values():
            metric.to(self.device)
        self.train_loss.to(self.device)
        self.val_loss.to(self.device)


    def forward(self, batch, apply_mod_dropout: bool = False):
        """
        Forward pass through the model.
        Args:
            batch: Input batch.
            apply_mod_dropout: Whether to apply modality dropout.
        
        Returns:
            dict_logits_task: Predictions for main tasks.
            dict_logits_aux: Predictions for auxiliary tasks.
        """
        return self.model(batch, apply_mod_dropout)


    def step(self, batch, training: bool = False):
        """
        Perform a forward pass and compute loss.
        
        Args:
            batch: Input batch.
            training: Whether the model is in training mode.
        
        Returns:
            loss_sum: Total computed loss.
            all_preds: Predictions for each task.
            all_targets: Ground truth labels.
        """
        # Apply modality dropout only in training
        apply_mod_dropout = self.mod_dropout if training else False

        dict_logits_task, dict_logits_aux = self.forward(batch, apply_mod_dropout)

        loss_sum = 0
        all_preds, all_targets = {}, {}

        for task in dict_logits_task.keys():
            targets = batch[task].to(self.device)

            # Convert one-hot to class index if necessary
            if targets.ndim == 4:  # (B, num_classes, H, W)
                targets = torch.argmax(targets, dim=1)  # (B, H, W)

            main_logits = dict_logits_task[task] 

            aux_logits = None
            if task in dict_logits_aux:
                aux_logits = []
                for mod in self.aux_loss_modalities:
                    aux_key = f"aux_{mod}_{task}"
                    if aux_key in dict_logits_aux[mod]:
                        aux_logits.append(dict_logits_aux[mod][task])

            with torch.no_grad():
                main_proba = torch.softmax(main_logits, dim=1) 
                main_preds = torch.argmax(main_proba, dim=1) 

                if aux_logits:
                    aux_logits = torch.mean(torch.stack(aux_logits), dim=0) 

            # Compute main loss
            main_loss = self.criterion[task](main_logits, targets)
            if torch.isnan(main_loss).any() or torch.isinf(main_loss).any():
                print(f"NaN or Inf detected in main loss for task {task}")

            # Compute auxiliary loss
            aux_loss = 0
            if aux_logits is not None:
                aux_losses = []
                for mod in self.aux_loss_modalities:
                    aux_loss_mod = self.criterion[f"aux_{mod}_{task}"](aux_logits, targets)
                    aux_weight = self.config['modalities']['aux_loss_weight'].get(mod, 1.0)
                    aux_losses.append(aux_weight * aux_loss_mod)

                if aux_losses:
                    aux_loss = torch.mean(torch.stack(aux_losses))

                if torch.isnan(aux_loss).any() or torch.isinf(aux_loss).any():
                    print(f"NaN or Inf detected in auxiliary loss for task {task}")

            task_weight = self.config['labels_configs'][task].get('task_weight', 1.0)  
            weighted_loss = task_weight * (main_loss + aux_loss)
            loss_sum += weighted_loss

            all_preds[task] = main_preds 
            all_targets[task] = targets.to(torch.int32)

        return loss_sum, all_preds, all_targets


    def training_step(self, batch, batch_idx):
        loss, all_preds, all_targets = self.step(batch, training=True)  # Apply dropout

        self.train_loss.update(loss)

        for task in all_preds.keys():
            self.train_metrics[task].update(all_preds[task], all_targets[task])

        return loss


    def on_train_epoch_end(self):
        """Ensure metrics are logged and reset properly."""
        self.log("train_loss", self.train_loss.compute(), prog_bar=True, logger=True, sync_dist=True)

        # Log and reset all task metrics dynamically
        for task, metric in self.train_metrics.items():
            self.log(f"train_miou_{task}", metric.compute(), prog_bar=True, logger=True, sync_dist=True)
            metric.reset()

        self.train_loss.reset()


    def validation_step(self, batch, batch_idx):
        loss, all_preds, all_targets = self.step(batch, training=False)  # No dropout

        self.val_loss.update(loss)

        for task in all_preds.keys():
            self.val_metrics[task].update(all_preds[task], all_targets[task])
            self.val_iou[task].update(all_preds[task], all_targets[task])

        return loss


    def on_validation_epoch_end(self):
        """Logs validation loss, mIoU, and per-class IoUs with class numbers and names."""
        
        # Compute loss and metrics
        val_epoch_loss = self.val_loss.compute()
        
        # Log overall validation loss
        self.log(
            "val_loss",
            val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        # Log per-task metrics (e.g., mIoU) and per-class IoUs
        for task in self.val_metrics.keys():
            val_epoch_miou = self.val_metrics[task].compute()
            iou_per_class = self.val_iou[task].compute()
            
            # Log mean IoU for the task
            self.log(
                f"val_miou_{task}",
                val_epoch_miou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            # Get class names (as a dict {class_number: class_name})
            class_names_dict = self.config['labels_configs'][task]['value_name']

            # Iterate over per-class IoUs and log each with class number and name
            for class_number, iou in enumerate(iou_per_class):
                class_name = class_names_dict.get(class_number, f"class_{class_number}")  # Default to class_{number} if missing
                
                self.log(
                    f"val_iou_{task}_{class_number}_{class_name}",
                    iou.item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )

            # Reset metrics for the next epoch
            self.val_metrics[task].reset()
            self.val_iou[task].reset()

        # Reset overall validation loss for the next epoch
        self.val_loss.reset()


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dict_logits_task, _ = self.forward(batch, apply_mod_dropout=False)  # No dropout
        
        prediction = {}

        for task in dict_logits_task.keys():
            proba = torch.softmax(dict_logits_task[task], dim=1)
            prediction[f"preds_{task}"] = torch.argmax(proba, dim=1)

        return prediction


    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        """
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",  # Scheduler step interval (epoch-based)
                    "monitor": "val_loss",  # Monitor validation loss for scheduler
                    "frequency": 1,  # Step every epoch
                    "strict": True,
                    "name": "Scheduler",  # Scheduler name
                },
            }
        return self.optimizer
