# to run the code python evaluation.py path/to/dataset.csv path/to/model_checkpoint.ckpt path/to/unique_chars.json path/to/maxlen.txt

import argparse
import pandas as pd
import torch
import os
import json
import numpy as np
from model import ConvLSTMCAMbiotic
from data_preprocessing import load_data, validate_columns, get_unique_chars, get_maxlen, smile_to_sequence, smiles_to_sequences, filter_smiles, pad_one_hot_sequences

def load_model(checkpoint_path):
    """load_model loads the trained model from the checkpoint path"""
    model = ConvLSTMCAMbiotic.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def validate_smiles(smiles, maxlen, unique_chars):
    """validate_smiles validates the SMILES string based on the criteria"""
    reasons = []
    if len(smiles) > maxlen:
        reasons.append("higher than maxlen")
    if any(char not in unique_chars for char in smiles):
        reasons.append("SMILES contain a character not indexed")
    return reasons

def process_dataset_for_evaluation(data, smiles_column, target_column, maxlen, unique_chars_file):
    """
    Process the dataset for evaluation to convert SMILES to one-hot encoding and pad them.
    """

    # Load the unique characters for char_to_index mapping
    with open(unique_chars_file, 'r') as file:
        smiles_chars = json.load(file)
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, int(maxlen))

    # Pad the one-hot encoded sequences
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, int(maxlen))

    # Extract labels
    labels = data[target_column].values if target_column in data.columns else None

    return padded_smiles, labels

def preprocess_and_filter_data(file_path, unique_chars_file, maxlen_file, smiles_column='SMILES', target_column='TARGET'):
    """Filter out data that does not meet the criteria and process the data for evaluation."""
    # Load unique characters and maxlen directly from files
    with open(unique_chars_file, 'r') as file:
        unique_chars = set(json.load(file))
    with open(maxlen_file, 'r') as file:
        maxlen = int(file.read().strip())

    # Load data
    data = load_data(file_path, [smiles_column, target_column])

    # Filter out SMILES strings based on criteria
    filtered_out_data = []
    valid_data = data.copy()
    for index, row in data.iterrows():
        reasons = validate_smiles(row[smiles_column], maxlen, unique_chars)
        if reasons:
            filtered_out_data.append((row[smiles_column], ', '.join(reasons)))
            valid_data.drop(index, inplace=True)

    if filtered_out_data:
        pd.DataFrame(filtered_out_data, columns=['SMILES', 'Reason']).to_csv('Invalid_smiles_not_evaluated.csv', index=False)
        print("Filtered SMILES saved to 'Invalid_smiles.csv'.")

    if valid_data.empty:
        raise ValueError("No valid SMILES strings found in the dataset.")

    # Process dataset for evaluation
    processed_data, labels = process_dataset_for_evaluation(valid_data, smiles_column, target_column, maxlen, unique_chars_file)

    return processed_data, labels





def evaluate(model, processed_data, labels):
    """Evaluate the model on the processed data."""
    with torch.no_grad():
        inputs = torch.tensor(processed_data).float().to(model.device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probabilities >= 0.8).astype(int) # standard should be 0.5 but it seems to be only predicting 1s so I changed it to 0.8
        return zip(labels, probabilities, predictions)

def main(dataset_path, checkpoint_path, unique_chars_file, maxlen_file):
    #check if files exist
    print(f"Dataset Path: {dataset_path}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Unique Chars File: {unique_chars_file}")
    print(f"Maxlen File: {maxlen_file}")
    print(f"Current Working Directory: {os.getcwd()}")
        # Check if files exist
    if not os.path.exists(unique_chars_file):
        raise FileNotFoundError(f"The unique characters file does not exist: {unique_chars_file}")
    if not os.path.exists(maxlen_file):
        raise FileNotFoundError(f"The maxlen file does not exist: {maxlen_file}")


    model = load_model(checkpoint_path)
    data = load_data(dataset_path, required_columns=['SMILES'])  # Assuming 'SMILES' is the minimum required column
    # Process and filter the data
    processed_data, labels = preprocess_and_filter_data(dataset_path, unique_chars_file, maxlen_file, 'SMILES', 'TARGET')
    evaluation_results = evaluate(model, processed_data, labels)

    results_df = pd.DataFrame(evaluation_results, columns=['TARGET', 'Predicted_Probability', 'Final_Prediction'])
    test_df = pd.read_csv(f'{dataset_path}')
    results_df = pd.concat([test_df[['COMPOUND_ID', 'SMILES']], results_df], axis=1)
    results_df.to_csv('evaluation_results.csv', index=False)
    print("Evaluation completed. Results saved to 'evaluation_results.csv'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a new dataset using the trained model.')
    parser.add_argument('dataset_path', type=str, help='The file path to the new dataset CSV file.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the trained model checkpoint.')
    parser.add_argument('unique_chars_file', type=str, help='Path to the JSON file of unique characters.')
    parser.add_argument('maxlen_file', type=str, help='Path to the text file containing the maxlen value.')
    args = parser.parse_args()

    main(args.dataset_path, args.checkpoint_path, args.unique_chars_file, args.maxlen_file)
