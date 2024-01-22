import argparse
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from model import ConvLSTMCAMbiotic
from data_loader import SMILESDataModule
import pandas as pd
import torch
import json
import datetime
import psutil
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(name)-30s] %(message)s ",
    handlers=[logging.StreamHandler()],
    level=logging.INFO)

def get_vocab_size(unique_chars_file):
    """get_vocab_size is the addition of the number of unique characters in the SMILES strings and 1 for the padding character"""
    with open(unique_chars_file, 'r') as file:
        unique_chars = json.load(file)
    return len(unique_chars) + 1

def evaluate_model(model, datamodule, base_file_name):
    """evaluate_model evaluates the model on the test dataset and saves the results to a CSV file"""
    model.eval()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    results = []
    for batch in datamodule.test_dataloader():
        inputs, targets = batch
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
        results.extend(zip(targets.cpu().numpy(), probabilities, predictions))

    results_df = pd.DataFrame(results, columns=['TARGET', 'Predicted_Probability', 'Final_Prediction'])
    test_df = pd.read_csv(f'{base_file_name}_test_labels.csv')
    results_df = pd.concat([test_df[['COMPOUND_ID', 'SMILES']], results_df], axis=1)
    results_df.to_csv(f'{base_file_name}_evaluation_results.csv', index=False)

def train_model(args, vocab_size):
    """train_model trains the model and saves the best checkpoint"""
    base_file_name = args.data_path.rsplit('.', 1)[0]
    unique_chars_file = f"{base_file_name}_unique_chars.json"
    vocab_size = get_vocab_size(unique_chars_file) 
    model = ConvLSTMCAMbiotic(base_filename=base_file_name, 
                              num_cnn_layers=args.cnn_layers, 
                              num_lstm_layers=args.lstm_layers, 
                              hidden_dim=args.lstm_units, 
                              output_size=args.output_size, 
                              learning_rate=args.learning_rate)

    ConvLSTMCAMbiotic.save_hyperparameters_to_json(model.hparams, filepath=f"{base_file_name}_model_params.json")
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=1,
                                          mode='min',
                                          dirpath='models/')
   
    data_module = SMILESDataModule(base_file_path='.', base_file_name=base_file_name, batch_size=args.batch_size)

    # Create tb_logs directory if it doesn't exist
    tb_logs_dir = "tb_logs"
    if not os.path.exists(tb_logs_dir):
        os.makedirs(tb_logs_dir)
    # Logger setup
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Initialize the trainer and fit the model
    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger)
    trainer.fit(model, datamodule=data_module)

    # Load the best model and test
    model = ConvLSTMCAMbiotic.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=data_module)

    # Evaluate the model and save the results
    evaluate_model(model, data_module, base_file_name)

def main(file_path):
    """main is the main function"""
    # Memory and runtime logging
    startTime = datetime.datetime.now()
    startmem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info("Script started.")
    logger.info(f"Initial memory usage: {startmem:.1f} MB")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    #check if CUDA is available
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    #parser arguments
    parser = ArgumentParser(description='Train the model on molecular data.')
    parser.add_argument('data_path', type=str, help='The file path to the CSV data file.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--cnn_filters', type=int, default=64, help='Number of CNN filters')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--cnn_layers', type=int, default=5, help='Number of CNN layers')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()
    #extract base file name and determine vocabulari size
    base_file_name = args.data_path.rsplit('.', 1)[0]
    unique_chars_file = f"{base_file_name}_unique_chars.json"
    vocab_size = get_vocab_size(unique_chars_file)    
    #train the model
    train_model(args, vocab_size)
    #end time and memory usage
    endTime = datetime.datetime.now()
    endmem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 # more accurate than 1000000
    logger.info(f"Script ended. Time taken: {endTime - startTime}")
    logger.info(f"Final memory usage: {endmem:.1f} MB")

if __name__ == '__main__':
    main(sys.argv[1])
