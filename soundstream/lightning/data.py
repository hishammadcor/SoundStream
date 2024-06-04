from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from lightning.data import Maestro
import torch.nn as nn
import os

# Define the LightningData class inheriting from LightningDataModule
class LightningData(LightningDataModule):
    def __init__(
        self,
        path: str = None,
        batch_size: int = 16,
        segment_time: int = 3,
        sampling_rate: int = 24000,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters

    def setup(self, stage=None):
        # Define common arguments for dataset creation
        factory_kwargs = {
            "segment_time": self.hparams.segment_time,
            "sampling_rate": self.hparams.sampling_rate,
        }

        # Setup for training stage
        if stage == "fit":
            if self.hparams.path is not None:
                # Initialize the training dataset
                train_dataset = Maestro(
                    path=self.hparams.path, split="train", **factory_kwargs
                )
                self.train_dataset = train_dataset
            else:
                raise ValueError("Please provide the path to the dataset")
        
        # Setup for validation stage
        if stage == "validate" or stage == "fit":
            if self.hparams.path is not None:
                # Initialize the validation dataset
                valid_dataset = Maestro(
                    path=self.hparams.path, split="valid", **factory_kwargs
                )
                self.valid_dataset = valid_dataset

    def train_dataloader(self):
        # Return the DataLoader for the training dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,  # Use the custom collate function
            num_workers=os.cpu_count(),  # Use all available CPU cores
        )

    def val_dataloader(self):
        # Return the DataLoader for the validation dataset
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,  # Use the custom collate function
            num_workers=os.cpu_count(),  # Use all available CPU cores
        )

    def collate_fn(self, batch):
        # Custom collate function to pad sequences to the same length
        lengths = torch.tensor([elem.shape[-1] for elem in batch])  # Get lengths of sequences
        return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths  # Pad sequences and return with lengths
