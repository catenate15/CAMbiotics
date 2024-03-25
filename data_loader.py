# data_loader.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMILESDataset(Dataset):
    def __init__(self, features_file, labels_file):
        logger.info(f"Loading features from {features_file}")
        self.features = np.load(features_file)
        logger.info(f"Loading labels from {labels_file}")
        self.labels = pd.read_csv(labels_file)['TARGET'].values
        logger.info(f"Dataset Loaded successfully with {len(self.features)} SMILES strings with {len(self.labels)} labels.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Keep dtype=torch.long for classification
        return features, label




class SMILESDataModule(pl.LightningDataModule):
    def __init__(self, base_file_path, base_file_name, batch_size=32):
        super().__init__()
        self.base_file_path = base_file_path
        # No need to prefix with 'standardized_' as files are directly named in data_preprocessing2.py
        self.base_file_name = base_file_name
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load the datasets
        self.train_dataset = SMILESDataset(
            os.path.join(self.base_file_path, f'{self.base_file_name}_train_features.npy'),
            os.path.join(self.base_file_path, f'{self.base_file_name}_train_labels.csv')
        )
        self.val_dataset = SMILESDataset(
            os.path.join(self.base_file_path, f'{self.base_file_name}_valid_features.npy'),
            os.path.join(self.base_file_path, f'{self.base_file_name}_valid_labels.csv')
        )
        self.test_dataset = SMILESDataset(
            os.path.join(self.base_file_path, f'{self.base_file_name}_test_features.npy'),
            os.path.join(self.base_file_path, f'{self.base_file_name}_test_labels.csv')
        )



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0) # shuffle is true because we want to shuffle the data during training to avoid the model from learning the order of the data

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)


if __name__ == '__main__':
    parser = ArgumentParser(description='SMILES Data Loading')
    parser.add_argument('--base_file_path', type=str, required=True, help='Base file path for the dataset files.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading.')

    args = parser.parse_args()

    data_module = SMILESDataModule(base_file_path=args.base_file_path, batch_size=args.batch_size)
    data_module.setup(stage='fit') 

    logger.info("Iterating over training batches:")
    for i, batch in enumerate(data_module.train_dataloader()):
        inputs, labels = batch
        logger.info(f"Batch {i}: tensor type {inputs.dtype}, Batch size: {len(inputs)}")
        if i == 2:
            break


