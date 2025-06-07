import torch

from torch import nn
from typing import Optional, Dict, Any

from flair_hub.models.flair_model import FLAIR_INC_Model
from flair_hub.data.datamodule import FlairDataModule
from flair_hub.tasks.tasks_module import SegmentationTask


    
def build_data_module(
    config: Dict[str, Any],
    dict_train: Optional[dict] = None,
    dict_val: Optional[dict] = None,
    dict_test: Optional[dict] = None,
) -> "FlairDataModule":
    """
    This function creates a data module for training, validation, and testing.
    Args:
        config (dict): Configuration dictionary containing parameters for the data module.
        dict_train (dict): Dictionary containing training data.
        dict_val (dict): Dictionary containing validation data.
        dict_test (dict): Dictionary containing test data.
    Returns:
        dm: Data module with specified configuration.
    """
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(config['modalities']['pre_processings']["use_augmentation"], bool), \
        "use_augmentation must be a boolean"


    dm = FlairDataModule(
        config=config,
        dict_train=dict_train,
        dict_val=dict_val,
        dict_test=dict_test,
        batch_size=config['hyperparams']["batch_size"],
        num_workers=config['hardware']["num_workers"],
        drop_last=True,
        use_augmentations=config['modalities']['pre_processings']["use_augmentation"],
    )

    return dm  



def build_segmentation_module(
    config: Dict[str, Any],
    in_img_sizes: Any,
    stage: str = 'train'
) -> "SegmentationTask":
    """
    This function creates a segmentation module for training or prediction.
    Args:
        config (dict): Configuration dictionary containing parameters for the segmentation module.
        in_img_sizes (Any): Input image sizes for the segmentation model.
        stage (str): Stage for which the segmentation module is created ('train' or 'predict').
    Returns:
        seg_module: Segmentation module with specified configuration.
    """
    assert stage in ['train', 'predict'], "stage must be either 'train' or 'predict'"

    model = FLAIR_INC_Model(config, in_img_sizes)

    if stage == 'train':
        flair_losses = FLAIRLosses(config)
        losses = flair_losses.get_losses()

        seg_module = SegmentationTask(
            model=model,
            config=config,
            criterion=losses,
        )

    elif stage == 'predict':
        seg_module = SegmentationTask(
            model=model,
            config=config,
        )

    return seg_module



def get_input_img_sizes(config: Dict[str, Any], dm: Any, stage: str = "fit") -> Dict[str, int]:
    """
    Retrieves input image sizes from the data module, handling both training and prediction modes.
    Args:
        config (dict): Configuration dictionary containing input modality information.
        dm (Any): Data module object containing the data loader.
        stage (str): One of "fit" or "predict" to indicate the mode.
    Returns:
        img_input_sizes (dict): Dictionary with modality names as keys and corresponding image sizes as values.
    """
    assert stage in {"fit", "predict"}, f"Unsupported stage '{stage}'"

    dm.setup(stage)

    if stage == "fit":
        dataloader = dm.train_dataloader()
    elif stage == "predict":
        dataloader = dm.predict_dataloader()

    monkeybatch = next(iter(dataloader))

    img_input_sizes = {}
    for modality, is_input in config['modalities']['inputs'].items():
        if is_input and modality in monkeybatch:
            img_input_sizes[modality] = monkeybatch[modality][0].shape[-1]

    return img_input_sizes






class FLAIRLosses:
    def __init__(self, config: Dict[str, dict]) -> None:
        """
        Initialize the loss module by creating a dictionary of losses for main tasks and auxiliary tasks.
        Args:
            config (dict): Configuration dictionary containing task labels and auxiliary losses.
        """
        self.config = config
        self.default_weights: Dict[str, torch.FloatTensor] = {}  # Store default class weights
        self.losses: nn.ModuleDict = self._build_losses()

    def _build_losses(self) -> nn.ModuleDict:
        """
        Constructs a ModuleDict of losses for all tasks and their auxiliary modalities.
        Returns:
            nn.ModuleDict: A dictionary containing losses for each task and auxiliary loss.
        """
        losses = nn.ModuleDict()

        for task in self.config['labels']:
            task_config = self.config['labels_configs'][task]
            losses[task] = self._create_task_loss(task, task_config)

            # Auxiliary losses: check both aux_loss and inputs are active
            for modality, aux_active in self.config['modalities']['aux_loss'].items():
                if aux_active and self.config['modalities']['inputs'].get(modality, False):
                    aux_loss_name = f"aux_{modality}_{task}"
                    losses[aux_loss_name] = self._create_aux_loss(task, modality, task_config)

        return losses

    def _create_task_loss(self, task_name: str, task_config: Dict[str, dict]) -> nn.Module:
        """
        Creates the main loss function for a task.
        Args:
            task_name (str): Name of the task.
            task_config (dict): Task-specific configuration.
        Returns:
            nn.Module: The loss function for the task.
        """
        default_w = self._compute_default_weights(task_config)
        self.default_weights[task_name] = default_w  # Store default weights for later retrieval
        return nn.CrossEntropyLoss(weight=default_w)

    def _create_aux_loss(self, task_name: str, modality: str, task_config: Dict[str, dict]) -> nn.Module:
        """
        Creates an auxiliary loss function for a given task and modality.
        Args:
            task_name (str): Name of the task.
            modality (str): Name of the modality.
            task_config (dict): Task-specific configuration.
        Returns:
            nn.Module: The auxiliary loss function.
        """
        modality_weights = self.default_weights[task_name].clone()
        modality_exceptions = task_config['value_weights'].get('per_modality_exceptions', {}).get(modality, None)

        # Check if modality_exceptions is not None and contains values
        if modality_exceptions:
            for key, value in modality_exceptions.items():
                modality_weights[key] = value

        return nn.CrossEntropyLoss(weight=modality_weights)

    def _compute_default_weights(self, task_config: Dict[str, dict]) -> torch.FloatTensor:
        """
        Computes the default weight tensor for a task.
        Args:
            task_config (dict): Task-specific configuration.
        Returns:
            torch.FloatTensor: The default class weights tensor.
        """
        default_w = torch.FloatTensor(
            [task_config['value_weights']['default']] * len(task_config['value_name'])
        )

        # Check if 'default_exceptions' exists and is not empty
        if 'default_exceptions' in task_config['value_weights'] and task_config['value_weights']['default_exceptions']:
            for key, value in task_config['value_weights']['default_exceptions'].items():
                default_w[key] = value

        return default_w

    def get_losses(self) -> nn.ModuleDict:
        """
        Returns the constructed ModuleDict of losses.
        Returns:
            nn.ModuleDict: The loss dictionary.
        """
        return self.losses

    def get_default_weights(self, task_name: Optional[str] = None) -> torch.FloatTensor:
        """
        Retrieves the default weights for a given task or all tasks if task_name is None.
        Args:
            task_name (str, optional): The task name to get weights for. Defaults to None.
        Returns:
            torch.FloatTensor or dict: The weight tensor for a specific task or all stored weights.
        """
        return self.default_weights if task_name is None else self.default_weights.get(task_name, None)





