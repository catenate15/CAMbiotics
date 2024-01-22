

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryCohenKappa, BinaryMatthewsCorrCoef, BinarySpecificity, BinaryAUROC
import json

logger = logging.getLogger(__name__)
# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConvLSTMCAMbiotic(pl.LightningModule):
    """
    ConvLSTMCAMbiotic is a PyTorch Lightning Module that contains the model architecture and the training loops.
    """
    def __init__(self, base_filename, num_cnn_layers=5, num_lstm_layers=2, hidden_dim=64, output_size=1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Load unique characters
        unique_chars_file = f"{base_filename}_unique_chars.json"
        with open(unique_chars_file, 'r') as file:
            unique_chars = json.load(file)

        # Number of unique characters
        num_unique_chars = len(unique_chars)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=num_unique_chars if i == 0 else hidden_dim,
                          out_channels=hidden_dim, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(num_features=hidden_dim)
            ) for i in range(num_cnn_layers)
        ])

        # LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            batch_first=True, num_layers=num_lstm_layers)

        # Fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)
        
        
        # Metric collection
        metrics = MetricCollection({
            'accuracy': BinaryAccuracy(),
            'f1_score': BinaryF1Score(),
            'cohen_kappa': BinaryCohenKappa(),
            'mcc': BinaryMatthewsCorrCoef(),
            'specificity': BinarySpecificity(),
            'auroc':  BinaryAUROC()
        })

        # Metric collection for training
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    @staticmethod
    def save_hyperparameters_to_json(hparams, filepath):
        # Check if hparams is an instance of AttributeDict or a similar object
        # If yes, convert it to a dictionary
        if not isinstance(hparams, dict):
            hparams = vars(hparams)

        # Save the dictionary to a JSON file
        with open(filepath, 'w') as f:
            json.dump(hparams, f, indent=4)


    
    def forward(self, x):
        # Permute the tensor to have the shape [batch_size, num_channels, seq_len]
        x = x.permute(0, 2, 1)

        # Apply each convolutional layer
        for conv in self.convs:
            x = conv(x)

        # LSTM layer(s)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use only the last output of the LSTM
        
        # Fully connected layer
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze()  # Sigmoid activation function used since it is binary classification
        return x





    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def compute_metrics(self, y_hat, y, metrics):
        y_hat_sigmoid = torch.sigmoid(y_hat)
        return metrics(y_hat_sigmoid, y.int())
    
    def step(self, batch, batch_idx, metrics):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        # Print the shape of y_hat for the first 5 batches
        if batch_idx < 5:
            print(f"Batch {batch_idx} - y_hat shape before squeeze:", y_hat.shape)


        # If y_hat has more than 1 dimension and the second dimension is 1, squeeze it.
        if y_hat.dim() > 1 and y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log_dict(self.compute_metrics(y_hat, y, metrics), on_step=True, on_epoch=True)
        self.log(f'{metrics.prefix}loss', loss, on_step=True, on_epoch=True)
        return loss


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.test_metrics)
