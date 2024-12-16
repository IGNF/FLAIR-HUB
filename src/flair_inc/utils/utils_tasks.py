import albumentations as A
import torch

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def get_segmentation_module(config, stage: str = 'train'):
    """
    This function creates a segmentation module for training or prediction.
    Parameters:
    config (dict): Configuration dictionary containing parameters for the segmentation module.
    stage (str): Stage for which the segmentation module is created ('train' or 'predict').
    Returns:
    seg_module: Segmentation module with specified configuration.
    """
    assert stage in ['train', 'predict'], "stage must be either 'train' or 'predict'"
                   
    model = FLAIR_TimeTexture(config)

    if stage == 'train':

        losses = nn.ModuleDict({})

        if 'AERIAL_LABEL-COSIA' in config['labels']: #TODO add multitask losses

            default_w = torch.FloatTensor([config['labels_configs']['AERIAL_LABEL-COSIA']['value_weights']['default']] * len(config['labels_configs']['AERIAL_LABEL-COSIA']['value_name']))
            for key, value in config['labels_configs']['AERIAL_LABEL-COSIA']['value_weights']['default_exceptions'].items():
                default_w[key] = value
            losses['AERIAL_LABEL-COSIA'] = nn.CrossEntropyLoss(weight=default_w)

            for modality, is_active in config['modalities']['aux_loss'].items():
                if is_active:
                    modality_weights = default_w.clone()
                    modality_exceptions = config['labels_configs']['AERIAL_LABEL-COSIA']['value_weights']['per_modality_exceptions'].get(modality)
                    if modality_exceptions:
                        for key, value in modality_exceptions.items():
                            modality_weights[key] = value
                    losses[modality] = nn.CrossEntropyLoss(weight=modality_weights)

 
        optimizer = torch.optim.SGD(model.parameters(), lr=config['hyperparams']["learning_rate"])

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=4,
            min_lr=1e-7,
        )

        seg_module = SegmentationTask(
            model=model,
            config=config,
            class_weights=default_w,
            criterion=losses,
            optimizer=optimizer,
            scheduler=scheduler,
            use_metadata=config['modalities']['pre_processings']["use_metadata"],
        )

    elif stage == 'predict':
        seg_module = SegmentationTask(
            model=model,
            config=config,
            use_metadata=config['modalities']['pre_processings']["use_metadata"],
        )        

    return seg_module  








