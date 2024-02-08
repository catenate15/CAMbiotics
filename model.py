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
    def __init__(self, base_filename, num_cnn_layers, num_lstm_layers, hidden_dim, output_size, learning_rate, vocab_size, attention_heads):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size  # Store vocab_size from arguments
        
        
        # Initialize dictionaries for activations and gradients
        self.activations = {}
        self.gradients = {}

        # Define convolutional layers dynamically with vocab_size
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=len(smiles_chars)  if i == 0 else hidden_dim,
                          out_channels=hidden_dim, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ) for i in range(num_cnn_layers)
        ])

        # LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_lstm_layers)

        # Multi-head attention layer
        self.attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
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
        
        
        # Register hooks
        self.register_hooks()

    def register_hooks(self):
        last_conv_layer = self.convs[-1][0]  # Accessing the actual convolutional layer if it's wrapped inside nn.Sequential
        last_conv_layer.register_forward_hook(self.get_activation_hook('last_conv'))
        last_conv_layer.register_full_backward_hook(self.get_gradient_hook('last_conv'))

    
    def get_activation_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def get_gradient_hook(self, name):
        def hook(model, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook


    @staticmethod
    def save_hyperparameters_to_json(hparams, filepath):
        """Save the hyperparameters to a JSON file."""
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        with open(filepath, 'w') as f:
            json.dump(hparams, f, indent=4)


    @property
    def last_conv_layer(self):
        return self.convs[-1]

    def forward(self, x):
        print(f"Initial input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # Adjusting input dimensions
        print(f"Input shape after permute for conv layers: {x.shape}")
    
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            print(f"Shape after conv layer {i}: {x.shape}")
    
        # After the loop, x contains the output of the last conv layer
        # Optionally set requires_grad to True for the last conv layer output for visualization
        x.requires_grad_(True)
        self.feature_maps = x
        self.feature_maps.retain_grad()  # Ensure gradients are retained for feature_maps
        #print(f"Feature maps shape (last conv layer output): {self.feature_maps.shape}")
    
        # Preparing for LSTM
        x = x.permute(0, 2, 1)  # Adjust dimensions for LSTM input
        print(f"Shape before LSTM: {x.shape}")
        x, _ = self.lstm(x)
        #print(f"Shape after LSTM: {x.shape}")
    
        # Applying Multi-head attention
        query = x.permute(1, 0, 2)
        attention_output, attention_weights = self.attention(query, query, query)
        attention_output = attention_output.permute(1, 0, 2)
        print(f"Attention output shape: {attention_output.shape}")
    
        # Applying the fully connected layer
        x = self.fc(attention_output[:, -1, :])
        #print(f"Output shape after FC layer: {x.shape}")
    
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

