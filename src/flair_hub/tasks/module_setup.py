import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule

from flair_hub.data.datamodule import FlairDataModule
from flair_hub.models.flair_model import FLAIR_HUB_Model
from flair_hub.tasks.tasks_module import SegmentationTask
from flair_hub.utils.class_config import model_class_indices, class_weights


def build_data_module(
    config: Dict[str, Any],
    dict_train: Optional[dict] = None,
    dict_val: Optional[dict] = None,
    dict_test: Optional[dict] = None,
) -> FlairDataModule:
    """
    Create the Lightning data module for training, validation, and prediction.
    """
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(
        config["modalities"]["pre_processings"]["use_augmentation"], bool
    ), "use_augmentation must be a boolean"

    return FlairDataModule(
        config=config,
        dict_train=dict_train,
        dict_val=dict_val,
        dict_test=dict_test,
        batch_size=config["hyperparams"]["batch_size"],
        num_workers=config["hardware"]["num_workers"],
        drop_last=True,
        use_augmentations=config["modalities"]["pre_processings"]["use_augmentation"],
    )


def build_segmentation_module(
    config: Dict[str, Any],
    in_img_sizes: Any,
    stage: str = "train",
) -> SegmentationTask:
    """
    Create the Lightning segmentation task module.

    Args:
        config: Full experiment configuration.
        in_img_sizes: Input spatial sizes inferred from the dataloader.
        stage: Either "train" or "predict".
    """
    assert stage in {"train", "predict"}, "stage must be either 'train' or 'predict'"

    model_config = deepcopy(config)
    resume_mode = config["tasks"]["train_tasks"].get(
        "resume_training_from_ckpt", False
    )
    checkpoint_path = config.get("paths", {}).get("ckpt_model_path")
    checkpoint_will_be_loaded = (
        stage == "predict"
        or config["tasks"]["train_tasks"].get("init_weights_only_from_ckpt", False)
        or resume_mode is True
        or (
            isinstance(resume_mode, str)
            and resume_mode.lower() == "auto"
            and bool(checkpoint_path)
            and os.path.isfile(checkpoint_path)
        )
    )
    if checkpoint_will_be_loaded:
        configured_weights = model_config["models"]["monotemp_model"].get(
            "encoder", {}
        ).get("weights")
        if configured_weights is not None:
            print(
                "[MODEL] Full checkpoint loading requested: encoder "
                "pretrained download disabled; architecture only."
            )
        model_config["models"]["monotemp_model"].setdefault(
            "encoder", {}
        )["weights"] = None

    model = FLAIR_HUB_Model(model_config, in_img_sizes)

    if stage == "train":
        losses = FLAIRLosses(config).get_losses()
        return SegmentationTask(
            model=model,
            config=config,
            criterion=losses,
        )

    return SegmentationTask(
        model=model,
        config=config,
    )


class ProbabilityEnsembleTask(LightningModule):
    """Prediction-only ensemble that averages member class probabilities."""

    def __init__(
        self,
        members: list[SegmentationTask],
        weights: list[float],
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        if len(members) < 2 or len(members) != len(weights):
            raise ValueError("An ensemble requires matching lists of at least two members")
        if any(weight <= 0 for weight in weights):
            raise ValueError("Ensemble weights must be positive")

        total = float(sum(weights))
        self.members = nn.ModuleList(members)
        self.weights = [float(weight) / total for weight in weights]
        self.tasks = list(config["labels"])
        self.output_code_buffers = {}

        first_codes = members[0].output_class_codes
        for member in members[1:]:
            if member.output_class_codes != first_codes:
                raise ValueError("All ensemble members must use identical output class codes")

        for index, task in enumerate(self.tasks):
            buffer_name = f"_ensemble_output_codes_{index}"
            self.register_buffer(
                buffer_name,
                torch.as_tensor(first_codes[task], dtype=torch.long),
                persistent=False,
            )
            self.output_code_buffers[task] = buffer_name

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        probability_sums = {}
        for member, weight in zip(self.members, self.weights):
            logits_by_task, _ = member.forward(batch, apply_mod_dropout=False)
            for task, logits in logits_by_task.items():
                weighted = float(weight) * F.softmax(logits, dim=1)
                if task in probability_sums:
                    probability_sums[task] = probability_sums[task] + weighted
                else:
                    probability_sums[task] = weighted

        return {
            f"preds_{task}": getattr(self, self.output_code_buffers[task])[
                torch.argmax(probabilities, dim=1)
            ]
            for task, probabilities in probability_sums.items()
        }


def get_input_img_sizes(
    config: Dict[str, Any],
    dm: Any,
    stage: str = "fit",
) -> Dict[str, int]:
    """
    Infer the spatial input size for each active input modality from one batch.

    Returns (height, width), preserving rectangular patch geometry.
    """
    assert stage in {"fit", "predict"}, f"Unsupported stage '{stage}'"

    dm.setup(stage)
    dataloader = dm.train_dataloader() if stage == "fit" else dm.predict_dataloader()
    monkeybatch = next(iter(dataloader))

    img_input_sizes = {}
    for modality, is_input in config["modalities"]["inputs"].items():
        if is_input and modality in monkeybatch:
            img_input_sizes[modality] = tuple(monkeybatch[modality].shape[-2:])

    return img_input_sizes


def _labels_to_class_indices(target: torch.Tensor) -> torch.Tensor:
    """
    Convert supported target formats to class-index labels.

    Supported target shapes:
        [B, C, H, W] one-hot/probability-like labels
        [B, 1, H, W]
        [B, H, W]
    """
    if target.dim() == 4:
        if target.size(1) == 1:
            target = target[:, 0]
        else:
            target = target.argmax(dim=1)

    return target.long()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the Lovasz extension with respect to sorted errors.

    Args:
        gt_sorted: Sorted foreground labels with shape [P], values 0 or 1.
    """
    num_pixels = gt_sorted.numel()
    total_foreground = gt_sorted.sum()

    intersection = total_foreground - gt_sorted.float().cumsum(0)
    union = total_foreground + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)

    if num_pixels > 1:
        jaccard[1:num_pixels] = jaccard[1:num_pixels] - jaccard[: num_pixels - 1]

    return jaccard


def flatten_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten probabilities and labels for Lovasz-Softmax.

    Args:
        probs: Predicted probabilities with shape [B, C, H, W].
        labels: Class-index labels with shape [B, H, W].
        ignore_index: Label value ignored by the loss.

    Returns:
        probs_flat: Flattened probabilities with shape [P, C].
        labels_flat: Flattened labels with shape [P].
    """
    if probs.dim() != 4:
        raise ValueError(f"Expected probs with shape [B, C, H, W], got {probs.shape}")

    _, num_classes, _, _ = probs.shape

    probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    labels_flat = labels.contiguous().view(-1)

    valid = labels_flat != ignore_index
    return probs_flat[valid], labels_flat[valid]


def lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss on flattened predictions.

    Args:
        probs: Flattened class probabilities with shape [P, C].
        labels: Flattened class-index labels with shape [P].
        classes: "present" to average only classes present in labels, or "all".
    """
    if probs.numel() == 0:
        return probs.sum() * 0.0

    if classes not in {"present", "all"}:
        raise ValueError("classes must be either 'present' or 'all'")

    losses = []
    num_classes = probs.size(1)

    for class_idx in range(num_classes):
        foreground = (labels == class_idx).float()

        if classes == "present" and foreground.sum() == 0:
            continue

        class_probs = probs[:, class_idx]
        errors = (foreground - class_probs).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]

        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground_sorted)))

    if not losses:
        return probs.sum() * 0.0

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovasz-Softmax loss.

    This loss accepts logits by default and supports one-hot or class-index targets.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        classes: str = "present",
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _labels_to_class_indices(target)
        probs = F.softmax(logits, dim=1) if self.from_logits else logits

        probs_flat, target_flat = flatten_probs(
            probs,
            target,
            ignore_index=self.ignore_index,
        )

        return lovasz_softmax_flat(
            probs_flat,
            target_flat,
            classes=self.classes,
        )


class OHEMCrossEntropyLoss(nn.Module):
    """Threshold-based online hard-example mining cross-entropy.

    The sampler selects valid pixels whose ground-truth class probability is
    below ``max(thresh, probability_at_min_kept)``. The selected per-pixel losses
    are divided by all B*H*W positions rather than only the selected positions.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        thresh: float = 0.7,
        min_kept: int = 100000,
    ) -> None:
        super().__init__()
        if min_kept <= 1:
            raise ValueError("OHEM min_kept must be greater than one")
        if not 0.0 <= thresh <= 1.0:
            raise ValueError("OHEM thresh must be in [0, 1]")

        self.register_buffer("weight", weight)
        self.ignore_index = int(ignore_index)
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)

    @torch.no_grad()
    def sample_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return the binary pixel-weight mask selected by online hard-example mining."""
        target = _labels_to_class_indices(target)
        if logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                "OHEM logits and targets must have the same spatial shape, "
                f"got {logits.shape[-2:]} and {target.shape[-2:]}"
            )

        valid_mask = target != self.ignore_index
        selected_mask = torch.zeros_like(valid_mask)
        if not valid_mask.any():
            return selected_mask

        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        probabilities = F.softmax(logits.detach().float(), dim=1)
        true_class_probabilities = probabilities.gather(
            1,
            safe_target.unsqueeze(1),
        ).squeeze(1)

        sorted_probabilities = true_class_probabilities[valid_mask].sort().values
        batch_kept = self.min_kept * target.shape[0]
        min_threshold = sorted_probabilities[
            min(batch_kept, sorted_probabilities.numel() - 1)
        ]
        threshold = torch.maximum(
            min_threshold,
            min_threshold.new_tensor(self.thresh),
        )
        selected_mask[valid_mask] = (
            true_class_probabilities[valid_mask] < threshold
        )
        return selected_mask

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _labels_to_class_indices(target)
        selected_mask = self.sample_mask(logits, target)
        per_pixel_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        return (per_pixel_loss * selected_mask).mean()


class CELovaszLoss(nn.Module):
    """
    Combined Cross-Entropy + Lovasz-Softmax loss for semantic segmentation.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        ignore_index: int = -100,
        ohem_enabled: bool = False,
        ohem_thresh: float = 0.7,
        ohem_min_kept: int = 100000,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight

        if ohem_enabled:
            self.ce = OHEMCrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_index,
                thresh=ohem_thresh,
                min_kept=ohem_min_kept,
            )
        else:
            self.ce = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_index,
            )
        self.lovasz = LovaszSoftmaxLoss(
            ignore_index=ignore_index,
            classes="present",
            from_logits=True,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _labels_to_class_indices(target)
        if not torch.any(target != -100):
            return logits.float().sum() * 0.0

        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            logits_fp32 = logits.float()
            ce_loss = self.ce(logits_fp32, target)
            lovasz_loss = self.lovasz(logits_fp32, target)

        return self.ce_weight * ce_loss + self.lovasz_weight * lovasz_loss


class FLAIRLosses:
    """
    Build the loss dictionary used by SegmentationTask.

    Main task loss:
        CE + Lovasz-Softmax

    Auxiliary losses:
        CE + Lovasz-Softmax, scaled by the configured auxiliary weight
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.default_weights: Dict[str, torch.FloatTensor] = {}
        self.losses = self._build_losses()

    def _build_losses(self) -> nn.ModuleDict:
        losses = nn.ModuleDict()

        for task in self.config["labels"]:
            task_config = self.config["labels_configs"][task]
            losses[task] = self._create_task_loss(task, task_config)

            for modality, aux_active in self.config["modalities"]["aux_loss"].items():
                input_active = self.config["modalities"]["inputs"].get(modality, False)

                if aux_active and input_active:
                    aux_loss_name = f"aux_{modality}_{task}"
                    losses[aux_loss_name] = self._create_aux_loss(
                        task_name=task,
                        modality=modality,
                        task_config=task_config,
                    )

        return losses

    def _create_task_loss(
        self,
        task_name: str,
        task_config: Dict[str, Any],
    ) -> nn.Module:
        default_weights = self._compute_default_weights(task_config)
        self.default_weights[task_name] = default_weights

        ohem_config = self._get_ohem_config(task_config)
        return CELovaszLoss(
            weight=default_weights,
            ce_weight=1.0,
            lovasz_weight=1.0,
            ignore_index=-100,
            **ohem_config,
        )

    def _create_aux_loss(
        self,
        task_name: str,
        modality: str,
        task_config: Dict[str, Any],
    ) -> nn.Module:
        modality_weights = self.default_weights[task_name].clone()
        modality_exceptions = (
            task_config["value_weights"]
            .get("per_modality_exceptions", {})
            .get(modality, None)
        )

        if modality_exceptions:
            output_indices = model_class_indices(task_config)
            raw_to_output = {
                raw_idx: output_idx
                for output_idx, raw_idx in enumerate(output_indices)
            }
            for class_idx, value in modality_exceptions.items():
                raw_idx = int(class_idx)
                if raw_idx in raw_to_output:
                    modality_weights[raw_to_output[raw_idx]] = value

        ohem_config = self._get_ohem_config(task_config)
        return CELovaszLoss(
            weight=modality_weights,
            ce_weight=1.0,
            lovasz_weight=1.0,
            ignore_index=-100,
            **ohem_config,
        )

    @staticmethod
    def _get_ohem_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
        ohem = task_config.get("loss", {}).get("ohem", {})
        return {
            "ohem_enabled": bool(ohem.get("enabled", False)),
            "ohem_thresh": float(ohem.get("thresh", 0.7)),
            "ohem_min_kept": int(ohem.get("min_kept", 100000)),
        }

    @staticmethod
    def _compute_default_weights(task_config: Dict[str, Any]) -> torch.FloatTensor:
        weights = class_weights(task_config)
        return torch.tensor([weights[code] for code in model_class_indices(task_config)],
                            dtype=torch.float32)

    def get_losses(self) -> nn.ModuleDict:
        return self.losses

    def get_default_weights(
        self,
        task_name: Optional[str] = None,
    ) -> Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]]:
        if task_name is None:
            return self.default_weights

        return self.default_weights.get(task_name)
