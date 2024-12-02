import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from flair_inc.data_loader import flair_dataset



def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)




def pad_collate_flair(sample_dict, pad_value=0):
    """
    Collate function for batching and padding. 
    Pads only the relevant SENTINEL fields and keeps other fields unchanged.
    Args:
    - sample_dict (list): List of dictionaries where each dictionary represents a sample.
    - pad_value (int): The value used for padding tensors.
    Returns:
    - dict: A dictionary with padded tensors and original string keys.
    """

    TO_PAD_KEYS = [
        'SENTINEL2_TS', 'SENTINEL2_TS_DATES',
        'SENTINEL1-ASC_TS', 'SENTINEL1-ASC_DATES',
        'SENTINEL1-DESC_TS', 'SENTINEL1-DESC_DATES'
    ]
    
    batch = {}
    
    for key in sample_dict[0].keys():
        if key in TO_PAD_KEYS:
            data = [i[key] for i in sample_dict]

            # Skip padding if all values are empty lists
            if all(len(e) == 0 for e in data):
                batch[key] = torch.empty((len(data), 0))  # Create an empty tensor with batch size
                continue

            # Proceed with padding
            sizes = [e.shape[0] for e in data if len(e) > 0]
            max_size = max(sizes) if sizes else 0

            padded_data = [
                pad_tensor(d, max_size, pad_value=pad_value) if len(d) > 0 else torch.zeros((max_size,), dtype=d.dtype) 
                for d in data
            ]
            batch[key] = torch.stack(padded_data, dim=0)

        elif isinstance(sample_dict[0][key], torch.Tensor):
            batch[key] = torch.stack([i[key] for i in sample_dict], dim=0)
        else:
            batch[key] = [i[key] for i in sample_dict]

    return batch



class flair_datamodule(LightningDataModule):

    def __init__(
        self, 
        config,
        dict_train : dict = None,
        dict_val : dict = None,
        dict_test : dict = None,
        num_workers : int = 1,
        batch_size : int = 2,
        drop_last : bool = True,
        use_augmentations : bool = True,
    ):
        
        super().__init__()
        self.config = config
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_augmentations = use_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = flair_dataset(
                self.config,
                dict_paths=self.dict_train,
                use_augmentations=self.use_augmentations,
            )

            self.val_dataset = flair_dataset(
                self.config,
                dict_paths=self.dict_val,
                use_augmentations=None,
            )

        elif stage == "predict":
            self.pred_dataset = flair_dataset(
                self.config,
                dict_paths=self.dict_test,
                use_augmentations=None,
            )
            

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn = pad_collate_flair
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn = pad_collate_flair
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn = pad_collate_flair
        )

    