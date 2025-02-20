import albumentations as A
import torch

from torch import nn
from flair_inc.models.flair_model import FLAIR_TimeTexture
from flair_inc.datamodule import FlairDataModule
from flair_inc.tasks_module import SegmentationTask


    
def get_data_module(config, 
                    dict_train : dict =None, 
                    dict_val : dict = None, 
                    dict_test : dict = None,
                    ):
    """
    This function creates a data module for training, validation, and testing.
    Parameters:
    config (dict): Configuration dictionary containing parameters for the data module.
    dict_train (dict): Dictionary containing training data.
    dict_val (dict): Dictionary containing validation data.
    dict_test (dict): Dictionary containing test data.
    Returns:
    dm: Data module with specified configuration.
    """
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(config['modalities']['pre_processings']["use_augmentation"], bool), "use_augmentation must be a boolean"
    assert isinstance(config['modalities']['pre_processings']["use_metadata"], bool), "use_metadata must be a boolean"   
    
    if config['modalities']['pre_processings']["use_augmentation"]:
        transform_set = A.Compose([A.VerticalFlip(p=0.5),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomRotate90(p=0.5)]
        )
    else:
        transform_set = None

    dm = FlairDataModule(
        config, 
        dict_train = dict_train,
        dict_val = dict_val,
        dict_test = dict_test,
        batch_size = config['hyperparams']["batch_size"],
        num_workers = config['hyperparams']["num_workers"],
        drop_last = True,
        use_augmentations = transform_set,
    )
    
    return dm    


def get_segmentation_module(config, in_img_sizes, stage: str = 'train'):
    """
    This function creates a segmentation module for training or prediction.
    Parameters:
    config (dict): Configuration dictionary containing parameters for the segmentation module.
    stage (str): Stage for which the segmentation module is created ('train' or 'predict').
    Returns:
    seg_module: Segmentation module with specified configuration.
    """
    assert stage in ['train', 'predict'], "stage must be either 'train' or 'predict'"
                   
    model = FLAIR_TimeTexture(config, in_img_sizes)

    if stage == 'train':

        flair_losses  = FLAIRLosses(config)
        losses = flair_losses.get_losses()
        default_weights = flair_losses.get_default_weights()

        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=config['hyperparams']["learning_rate"],
                                      weight_decay=config['hyperparams']['optim_weight_decay'],
                                      betas=tuple(config['hyperparams']['optim_betas']),
                    )    

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=4,
            min_lr=1e-7,
        )

        #scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #    optimizer,
        #    max_lr=config['hyperparams']["learning_rate"],
        #    total_steps=config['hyperparams']['steps'],
        #    pct_start=0.2,
        #    cycle_momentum=False,
        #    div_factor=1000,
        #)

        seg_module = SegmentationTask(
            model=model,
            config=config,
            criterion=losses,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    elif stage == 'predict':
        seg_module = SegmentationTask(
            model=model,
            config=config,
        )        

    return seg_module  



def get_input_img_sizes(config, dm):

    dm.setup("fit")
    monkeybatch = next(iter(dm.train_dataloader()))

    img_input_sizes = {}
    for k in config['modalities']['inputs']:
        if config['modalities']['inputs'][k]:
            img_input_sizes[k] = monkeybatch[k][0].shape[-1]  

    return img_input_sizes  



class FLAIRLosses:
    def __init__(self, config):
        """
        Initialize the loss module by creating a dictionary of losses for main tasks and auxiliary tasks.
        Args:
            config (dict): Configuration dictionary containing task labels and auxiliary losses.
        """
        self.config = config
        self.default_weights = {}  # Store default class weights
        self.losses = self._build_losses()

    def _build_losses(self):
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

    def _create_task_loss(self, task_name, task_config):
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

    def _create_aux_loss(self, task_name, modality, task_config):
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

    
    def _compute_default_weights(self, task_config):
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


    def get_losses(self):
        """
        Returns the constructed ModuleDict of losses.
        Returns:
            nn.ModuleDict: The loss dictionary.
        """
        return self.losses

    def get_default_weights(self, task_name=None):
        """
        Retrieves the default weights for a given task or all tasks if task_name is None.
        Args:
            task_name (str, optional): The task name to get weights for. Defaults to None.
        Returns:
            torch.FloatTensor or dict: The weight tensor for a specific task or all stored weights.
        """
        return self.default_weights if task_name is None else self.default_weights.get(task_name, None)




