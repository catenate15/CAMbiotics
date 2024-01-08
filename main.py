import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader import SMILESDataModule
from model import SMILESClassifier 



def main(args):
    # Initialize data module with dataset file path
    data_module = SMILESDataModule(base_file_path=args.dataset_file, batch_size=32)

    # Initialize model with parameters from args
    model = SMILESClassifier(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        cnn_filters=args.cnn_filters,
        lstm_units=args.lstm_units,
        output_size=args.output_size,
        learning_rate=args.learning_rate
    )

    # Ensure the models directory exists
    os.makedirs('models/', exist_ok=True) # Creates the models directory if it doesn't exist

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/', # Path to save the checkpoints
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # Initialize a trainer
    trainer = pl.Trainer(callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model on SMILES data')
    parser.add_argument('dataset_file', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--cnn_filters', type=int, default=64, help='Number of CNN filters')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()
    main(args)


