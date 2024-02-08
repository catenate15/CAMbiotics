# to run the code python evaluation.py path/to/evaluation_dataset.csv path/to/model_checkpoint.ckpt 

import argparse
import pandas as pd
import torch
import os
import json
import numpy as np
from model import ConvLSTMCAMbiotic
from data_preprocessing1 import standardize_smiles_column
from data_preprocessing2 import  load_data, validate_columns,  smile_to_sequence, smiles_to_sequences, filter_smiles, pad_one_hot_sequences
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Smile characters used in the model defined
smiles_chars = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']

maxlen = 350 # max length of which model was trained on

def load_model(checkpoint_path):
    """load_model loads the trained model from the checkpoint path"""
    model = ConvLSTMCAMbiotic.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def validate_smiles(smiles, maxlen, smiles_chars):
    """validate_smiles validates the SMILES string based on the criteria"""
    reasons = []
    if len(smiles) > maxlen:
        reasons.append("higher than maxlen")
    if any(char not in smiles_chars for char in smiles):
        reasons.append("SMILES contain a character not indexed")
    return reasons



def preprocess_and_filter_data(file_path, smiles_column='SMILES', target_column='TARGET'):
    """Filter out data that does not meet the criteria and process the data for evaluation."""
    # Load data
    data = load_data(file_path, [smiles_column, target_column])

    # Standardize the SMILES strings
    data = standardize_smiles_column(data, smiles_column)
    standardized_smiles_column = 'STANDARDIZED_SMILES'

    # Filter out SMILES strings based on criteria
    filtered_out_data = []
    valid_data = data.copy()
    for index, row in valid_data.iterrows():
        reasons = validate_smiles(row[standardized_smiles_column], maxlen, smiles_chars)
        if reasons:
            filtered_out_data.append((row[standardized_smiles_column], ', '.join(reasons)))
            valid_data.drop(index, inplace=True)

    if filtered_out_data:
        pd.DataFrame(filtered_out_data, columns=[standardized_smiles_column, 'Reason']).to_csv('Invalid_smiles_not_evaluated.csv', index=False)
        print("Filtered SMILES saved to 'Invalid_smiles_not_evaluated.csv'.")

    if valid_data.empty:
        raise ValueError("No valid SMILES strings found in the dataset.")

    # Process dataset for evaluation
    processed_data, labels = process_dataset_for_evaluation(valid_data, standardized_smiles_column, target_column, maxlen, smiles_chars)

    return processed_data, labels


def process_dataset_for_evaluation(data, smiles_column, target_column, maxlen, smiles_chars):
    """
    Process the dataset for evaluation to convert SMILES to one-hot encoding and pad them.
    """

    # Create the char_to_index mapping
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, maxlen)

    # Pad the one-hot encoded sequences
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, maxlen)

    # Extract labels if the target column exists in the dataset
    labels = data[target_column].values if target_column in data.columns else None

    return padded_smiles, labels




def evaluate(model, processed_data, labels):
    """Evaluate the model on the processed data."""
    with torch.no_grad():
        inputs = torch.tensor(processed_data).float().to(model.device)
        outputs_tuple = model(inputs)
        # Assuming the primary output (logits) is the first element of the tuple
        primary_output = outputs_tuple[0]
        probabilities = torch.sigmoid(primary_output).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
        return zip(labels, probabilities, predictions)


def main(dataset_path, checkpoint_path):
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Checkpoint Path: {checkpoint_path}")
    logger.info(f"Current Working Directory: {os.getcwd()}")

    # Check file existence
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    # Load model and data
    model = load_model(checkpoint_path)
    processed_data, labels = preprocess_and_filter_data(dataset_path, 'SMILES', 'TARGET')
    evaluation_results = evaluate(model, processed_data, labels)

    results_df = pd.DataFrame(evaluation_results, columns=['TARGET', 'Predicted_Probability', 'Final_Prediction'])
    test_df = pd.read_csv(dataset_path)
    results_df = pd.concat([test_df[['COMPOUND_ID', 'SMILES']], results_df], axis=1)

    # Check the DataFrame before saving
    logger.info(f"Results DataFrame:\n{results_df.head()}")

    # Convert 'Final_Prediction' from list to integer
    results_df['Final_Prediction'] = results_df['Final_Prediction'].apply(lambda x: x[0])

    # Add 'AGREE' column
    results_df['CONFIRMATION'] = results_df.apply(lambda row: 'AGREE' if row['TARGET'] == row['Final_Prediction'] else 'DISAGREE', axis=1)

    # Save the results
    results_file_path = 'evaluation_results.csv'
    results_df.to_csv(results_file_path, index=False)
    logger.info(f"Evaluation completed. Results saved to '{results_file_path}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a new dataset using the trained model.')
    parser.add_argument('dataset_path', type=str, help='The file path to the new dataset CSV file.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the trained model checkpoint.')
    args = parser.parse_args()

    # No longer need to pass smiles_chars_file and maxlen_file
    main(args.dataset_path, args.checkpoint_path)
