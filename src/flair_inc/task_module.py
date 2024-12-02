import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.aggregation import MeanMetric
import pytorch_lightning as pl
from typing import Dict, Any


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        config: Dict[str, Any],
        class_infos: dict,
        criterion=None,
        optimizer=None,
        use_metadata: bool = False,
        scheduler=None,
    ):
        """
        Initialize the Segmentation Task model.
        Args:
            model: The model used for segmentation.
            config (Dict[str, Any]): Configuration dictionary.
            class_infos (dict): Information about the classes, including weights and names.
            criterion: Loss function to optimize (default: None).
            optimizer: Optimizer to use for training (default: None).
            use_metadata (bool): Flag to indicate whether metadata is used (default: False).
            scheduler: Learning rate scheduler (default: None).
        """
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_metadata = use_metadata

        self.num_classes = len(class_infos)
        self.class_names = [class_infos[i][1] for i in class_infos]
        self.class_weights = [class_infos[i][0] for i in class_infos]

        self.train_metrics = MulticlassJaccardIndex(
            num_classes=self.num_classes, average='weighted'
        )
        self.val_metrics = MulticlassJaccardIndex(
            num_classes=self.num_classes, average='weighted'
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes, average=None
        )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()


    def setup(self, stage=None):
        """
        Setup datasets based on the stage.
        Args:
            stage (str, optional): The stage to setup for. Can be 'fit', 'validate', or 'predict'.
        """
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None


    def forward(self, batch):
        """
        Forward pass through the model.
        Args:
            batch (dict): Input batch for the model.
        Returns:
            logits (Tensor): Output from the model.
        """
        logits = self.model(batch)
        return logits


    def step(self, batch):
        """
        Perform a step of training or validation.
        Args:
            batch (dict): Input batch.
        Returns:
            loss (Tensor): Computed loss for the batch.
            preds (Tensor): Predictions for the batch.
            targets (Tensor): Ground truth labels for the batch.
        """
        logits = self.forward(batch)
        targets = batch['LABELS'].squeeze(2)

        targets = torch.argmax(targets, dim=1)
        loss = self.criterion(logits, targets)

        with torch.no_grad():
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            preds = preds.flatten(start_dim=1)
            targets = targets.flatten(start_dim=1).type(torch.int32)

        return loss, preds, targets


    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        Args:
            batch (dict): Input batch.
            batch_idx (int): The index of the current batch.
        Returns:
            loss (Tensor): Computed loss for the batch.
        """
        loss, preds, targets = self.step(batch)
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss


    def on_train_epoch_end(self):
        """
        Actions to perform at the end of the training epoch.
        Logs the training loss and metrics.
        """
        train_epoch_loss = self.train_loss.compute()
        train_epoch_metrics = self.train_metrics.compute()
        self.log(
            "train_loss",
            train_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        self.train_loss.reset()
        self.train_metrics.reset()


    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        Args:
            batch (dict): Input batch.
            batch_idx (int): The index of the current batch.
        Returns:
            loss (Tensor): Computed loss for the batch.
        """
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        self.val_iou(preds=preds, target=targets)
        return loss


    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        Logs the validation loss, mIoU, and per-class IoUs.
        """
        val_epoch_loss = self.val_loss.compute()
        val_epoch_metrics = self.val_metrics.compute()
        iou_per_class = self.val_iou.compute()

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
        self.log(
            "val_miou",
            val_epoch_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        for class_name, class_weight, iou in zip(self.class_names, self.class_weights, iou_per_class):
            if class_weight != 0:
                self.log(
                    f"val_iou_{class_name}",
                    iou.item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )

        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_iou.reset()


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Perform a prediction step.
        Args:
            batch (dict): Input batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader.
        Returns:
            batch (dict): Batch with predictions added.
        """
        logits = self.forward(batch)
        proba = torch.softmax(logits, dim=1)
        batch["preds"] = torch.argmax(proba, dim=1)
        return batch


    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Returns:
            dict: A dictionary containing the optimizer and scheduler configurations.
        """
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
                "name": "Scheduler"
            }
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            return self.optimizer
