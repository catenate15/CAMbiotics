import argparse
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import ConvLSTMCAMbiotic
from data_loader import SMILESDataModule
import torch
import json
import datetime
import psutil
import sys
import logging
from data_preprocessing2 import smiles_chars

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)-30s] %(message)s ", handlers=[logging.StreamHandler()], level=logging.INFO)



vocab_size = len(smiles_chars)
print(f"Vocab_size is {(vocab_size)}")

def set_seed(seed=500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_model(args, vocab_size, attention_heads):
    base_file_name = args.data_path.rsplit('.', 1)[0]

    # Initialize model with attention mechanism hyperparameters
    model = ConvLSTMCAMbiotic(
        num_cnn_layers=args.cnn_layers,
        num_lstm_layers=args.lstm_layers,
        hidden_dim=args.lstm_units,
        learning_rate=args.learning_rate,
        vocab_size=vocab_size,
        attention_heads=attention_heads,
        output_size=3  # Ensure this matches the number of classes
    )






    class FirstThreeCheckpoint(ModelCheckpoint):
        @property
        def state_key(self) -> str:
            return "FirstThreeCheckpoint"

    class LastThreeCheckpoint(ModelCheckpoint):
        @property
        def state_key(self) -> str:
            return "LastThreeCheckpoint"

    # Checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models500/',
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,  # Saves best 3 models based on validation loss
        mode='min',
        monitor='val_loss'  # Monitor validation loss for checkpointing
    )

    # Now use these in your trainer
    first_three_checkpoint = FirstThreeCheckpoint(
        dirpath='models500/first_three',
        filename='epoch={epoch:02d}',
        save_top_k=3,
        mode='min',
        monitor='val_loss'
    )

    last_three_checkpoint = LastThreeCheckpoint(
        dirpath='models500/last_three',
        filename='epoch={epoch:02d}',
        save_last=True,
        mode='min',
        monitor='val_loss'
    )

    # Data module setup
    data_module = SMILESDataModule(base_file_path='.', base_file_name=base_file_name, batch_size=args.batch_size)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=TensorBoardLogger("tb_logs500", name="my_model"),
        callbacks=[checkpoint_callback, first_three_checkpoint, last_three_checkpoint]
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


def main(file_path):
    # Set the seed for reproducibility
    set_seed(500) # change this to any integer to test robustness of model
    startTime = datetime.datetime.now()
    startmem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info("Script started.")
    logger.info(f"Initial memory usage: {startmem:.1f} MB")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    parser = ArgumentParser(description='Train the model on molecular data.')
    parser.add_argument('data_path', type=str, help='The file path to the CSV data file.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--cnn_filters', type=int, default=64, help='Number of CNN filters')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--cnn_layers', type=int, default=5, help='Number of CNN layers')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    args = parser.parse_args()
    base_file_name = args.data_path.rsplit('.', 1)[0]
    vocab_size = len(smiles_chars) + 1
    attention_heads = args.attention_heads # retrieve attention heads from args
    logger_info = (f"vocab_size is {(vocab_size)}")
    train_model(args, vocab_size, attention_heads)
    endTime = datetime.datetime.now()
    endmem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info(f"Script ended. Time taken: {endTime - startTime}")
    logger.info(f"Final memory usage: {endmem:.1f} MB")

if __name__ == '__main__':
    main(sys.argv[1])
