import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import (
    BinaryAccuracy,
    F1Score as BinaryF1Score,
    CohenKappa as BinaryCohenKappa,
    MatthewsCorrcoef as BinaryMatthewsCorrCoef,
    Specificity as BinarySpecificity,
    AUROC as BinaryAUROC
)

class SMILESClassifier(pl.LightningModule):
    def __init__(
        self, 
        vocab_size, # The number of unique characters in the SMILES alphabet
        embedding_dim=128, # The size of the embedding vectors
        cnn_filters=64, # The number of filters in the convolutional layers
        lstm_units=64, # The number of units in the LSTM layer
        output_size=1,# The number of units in the output layer
        learning_rate=1e-3 # The learning rate for the optimizer,  
    ):
        super(SMILESClassifier, self).__init__()
        
        # Embedding layer that turns tokens into embeddings
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=lstm_units, out_features=output_size)

         # Hyperparameters
        self.learning_rate = learning_rate


        # Metrics
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.cohen_kappa = BinaryCohenKappa()
        self.mcc = BinaryMatthewsCorrCoef()
        self.specificity = BinarySpecificity()
        self.auroc = BinaryAUROC()
        
    def forward(self, x, return_cam=False): # set return_cam to True if you want to return the feature maps if not set to false during model training
        # Pass the input through the embedding layer
        print(f"Input shape: {x.shape}")
        x = self.embedding(x)
        print(f"After embedding shape: {x.shape}")
        # Convolutional layers expect input in the shape (batch_size, channels, sequence_length)
        # so we permute the output from the embedding layer
        
        x = x.permute(0, 2, 1)
        print(f"After permute shape: {x.shape}")
        # Pass through convolutional layers with activation function and pooling
        x1 = F.relu(self.conv1(x)) #x1: The output of the first convolutional layer
        x2 = F.relu(self.conv2(x1)) #x2: The output of the second convolutional layer
        
        # LSTM layer
        x, _ = self.lstm(x2) #x: The output of the LSTM layer
        
        # We're using the output of the last time step for the fully connected layer
        x = self.fc(x[:, -1, :]) #x: The output of the fully connected layer
        
        # Apply a sigmoid activation function for binary classification
        x = torch.sigmoid(x) #x: The final output of the model
        
        if return_cam:
            # If return_cam is True, return both the prediction and the feature maps
            return x, x2
        else:
            return x

    def configure_optimizers(self):
        # Using the learning rate hyperparameter
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        # Log metrics
        self.log('train_accuracy', self.accuracy(y_hat, y.int()))
        self.log('train_f1_score', self.f1_score(y_hat, y.int()))
        self.log('train_cohen_kappa', self.cohen_kappa(y_hat, y.int()))
        self.log('train_mcc', self.mcc(y_hat, y.int()))
        self.log('train_specificity', self.specificity(y_hat, y.int()))
        self.log('train_auroc', self.auroc(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        # Log metrics
        self.log('val_accuracy', self.accuracy(y_hat, y.int()))
        self.log('val_f1_score', self.f1_score(y_hat, y.int()))
        self.log('val_cohen_kappa', self.cohen_kappa(y_hat, y.int()))
        self.log('val_mcc', self.mcc(y_hat, y.int()))
        self.log('val_specificity', self.specificity(y_hat, y.int()))
        self.log('val_auroc', self.auroc(y_hat, y))
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        # Log metrics
        self.log('test_accuracy', self.accuracy(y_hat, y.int()))
        self.log('test_f1_score', self.f1_score(y_hat, y.int()))
        self.log('test_cohen_kappa', self.cohen_kappa(y_hat, y.int()))
        self.log('test_mcc', self.mcc(y_hat, y.int()))
        self.log('test_specificity', self.specificity(y_hat, y.int()))
        self.log('test_auroc', self.auroc(y_hat, y))
        return loss
