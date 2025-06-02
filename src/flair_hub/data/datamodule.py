from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Dict, Optional

from src.flair_hub.data.dataloader import flair_dataset
from src.flair_hub.data.utils_data.padding import pad_collate_flair


class FlairDataModule(LightningDataModule):
    def __init__(
        self, 
        config,
        dict_train: Optional[Dict] = None,
        dict_val: Optional[Dict] = None,
        dict_test: Optional[Dict] = None,
        num_workers: int = 1,
        batch_size: int = 2,
        drop_last: bool = True,
        use_augmentations: bool = True,
    ):
        """
        Initialize the FlairDataModule.
        
        Args:
            config: Configuration dictionary with settings for the dataset.
            dict_train (dict, optional): Dictionary containing paths for training data.
            dict_val (dict, optional): Dictionary containing paths for validation data.
            dict_test (dict, optional): Dictionary containing paths for test data.
            num_workers (int, optional): Number of workers for data loading.
            batch_size (int, optional): Batch size for data loading.
            drop_last (bool, optional): Whether to drop the last batch in training and validation.
            use_augmentations (bool, optional): Whether to apply augmentations.
        """
        super().__init__()
        self.config = config
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.use_augmentations = use_augmentations
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None

    def prepare_data(self):
        """Prepare the data. This is a placeholder function."""
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for training, validation, or prediction.
        
        Args:
            stage (str, optional): The stage for setting up the data ('fit', 'validate', 'predict').
        """
        if stage in ("fit", "validate"):
            self._setup_train_val_datasets()
        elif stage == "predict":
            self._setup_pred_dataset()

    def _setup_train_val_datasets(self):
        """Helper function to set up the training and validation datasets."""
        self.train_dataset = self._create_flair_dataset(self.dict_train, self.use_augmentations)
        self.val_dataset = self._create_flair_dataset(self.dict_val, use_augmentations=None)

    def _setup_pred_dataset(self):
        """Helper function to set up the prediction dataset."""
        self.pred_dataset = self._create_flair_dataset(self.dict_test, use_augmentations=None)

    def _create_flair_dataset(self, dict_paths: Dict[str, str], use_augmentations: Optional[bool]) -> "flair_dataset":
        """
        Helper function to create a flair dataset.

        Args:
            dict_paths (dict): Dictionary containing the paths to the data.
            use_augmentations (bool, optional): Whether to apply augmentations.

        Returns:
            flair_dataset: The created dataset.
        """
        return flair_dataset(
            self.config,
            dict_paths=dict_paths,
            use_augmentations=use_augmentations
        )

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool, drop_last: bool):
        """
        Helper function to create a DataLoader for a given dataset.

        Args:
            dataset: The dataset to be used in the DataLoader.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.
            drop_last (bool): Whether to drop the last batch.

        Returns:
            DataLoader: The configured DataLoader.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            collate_fn=pad_collate_flair
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for training.
        """
        return self._create_dataloader(self.train_dataset, self.batch_size, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        """
        Returns the DataLoader for validation.
        """
        return self._create_dataloader(self.val_dataset, self.batch_size, shuffle=False, drop_last=self.drop_last)

    def predict_dataloader(self):
        """
        Returns the DataLoader for prediction.
        """
        return self._create_dataloader(self.pred_dataset, batch_size=1, shuffle=False, drop_last=False)



    