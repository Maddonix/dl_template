from typing import Optional, Tuple, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
# from src.datamodules.datasets.audio_dataset import AudioDataset
import sys

sys.path.append("/home/lux_t1/Desktop")
from data_store import AudioDatasetStrong

class AudioDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        classes: List,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 5,
        pin_memory: bool = False,
        sample_rate: int = 41000,
        duration: int = 10000,  # in ms
        **kwargs,
    ):
        super().__init__()
        self.classes = classes
        self.data_path = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dims = (1, self.duration * self.sample_rate)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = AudioDatasetStrong(self.sample_rate, self.classes, label_type ="weak")
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, [int(_ * len(dataset)) for _ in self.train_val_test_split]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
