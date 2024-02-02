import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryCohenKappa, BinaryMatthewsCorrCoef, BinarySpecificity, BinaryAUROC
import json
from data_preprocessing2 import smiles_chars
from torch.nn import MultiheadAttention

class ConvLSTMCAMbiotic(pl.LightningModule):
    """ConvLSTMCAMbiotic model for classification of SMILES strings with attention mechanism."""
    def __init__(self, base_filename, num_cnn_layers, num_lstm_layers, hidden_dim, output_size, learning_rate, vocab_size, attention_heads):
        super().__init__()
        self.save_hyperparameters()

        # Initialize variables for feature maps and gradients
        self.feature_maps = None
        self.gradients = None
        self.hook_registered = False

        # Number of unique characters in the SMILES vocabulary
        num_smiles_chars = len(smiles_chars)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=num_smiles_chars if i == 0 else hidden_dim,
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

        # Multi-head attention layer
        self.attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads)

        # Register the hook in the constructor
        self.register_backward_hook(self.save_gradients)

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
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def save_gradients(self, module, grad_input, grad_output):
        """Save the gradients for the feature maps."""
        if self.feature_maps.requires_grad:
            self.gradients = grad_output[0]

    @staticmethod
    def save_hyperparameters_to_json(hparams, filepath):
        """Save the hyperparameters to a JSON file."""
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        with open(filepath, 'w') as f:
            json.dump(hparams, f, indent=4)




    def forward(self, x):
        # Print the input shape
        print(f"Input shape at the start of forward: {x.shape}")

        # Permute the tensor to have the shape [batch_size, num_channels, seq_len]
        x = x.permute(0, 2, 1)

        # Print the shape after permutation
        print(f"Input shape after permutation: {x.shape}")

        # Apply convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            # Check if it's the last convolutional layer
            if i == len(self.convs) - 1:
                # Set requires_grad to True for the output of the last conv layer
                x.requires_grad_(True)
                # Assign the output to self.feature_maps
                self.feature_maps = x
                #retains the gradient for self.feature_maps because they are a non-leaf tensor and are not retained by default
                self.feature_maps.retain_grad() 
                # Print the status of requires_grad for self.feature_maps
                print(f"self.feature_maps.requires_grad: {self.feature_maps.requires_grad}")

        

        # Permute the tensor for LSTM input
        x = self.feature_maps.permute(0, 2, 1)

        # Apply LSTM layers
        x, _ = self.lstm(x)

        # Take only the last output of the LSTM
        x = x[:, -1, :]

        # Apply attention layer
        attention_output, attention_weights = self.attention(x, x, x)

        # Fully connected layer
        x = self.fc(attention_output.squeeze(1) if attention_output.dim() == 3 else attention_output)

        # Return output, attention weights, and feature maps
        return x, attention_weights, self.feature_maps


    def register_gradient_hook(self):
        """Register gradient hook for the feature maps."""
        if self.feature_maps is not None and self.feature_maps.requires_grad:
            if not self.hook_registered:
                self.hook = self.feature_maps.register_hook(self.save_gradients)
                self.hook_registered = True
        else:
            print("Warning: Feature maps tensor does not require gradients or is not set.")


    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def compute_metrics(self, y_hat, y, metrics):
        """Compute and return the metrics for the model."""
        y_hat_sigmoid = torch.sigmoid(y_hat)
        return metrics(y_hat_sigmoid, y.int())

    def step(self, batch, batch_idx, metrics):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Call forward and extract the actual output (ignoring attention_weights and feature_maps)
        y_hat, _, _ = self(x)  # Unpack the tuple

        # Check and process y_hat as before
        if y_hat.dim() > 1 and y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Compute metrics
        metric_results = self.compute_metrics(y_hat, y, metrics)
        for key, value in metric_results.items():
            self.log(key, value, on_step=True, on_epoch=True)

        self.log(f'{metrics.prefix}loss', loss, on_step=True, on_epoch=True)
        return loss


    def training_step(self, batch, batch_idx):
        """Define the training step."""
        return self.step(batch, batch_idx, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        """Define the validation step."""
        return self.step(batch, batch_idx, self.val_metrics)

    def test_step(self, batch, batch_idx):
        """Define the test step."""
        return self.step(batch, batch_idx, self.test_metrics)

