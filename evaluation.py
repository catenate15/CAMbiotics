# to run the code python evaluation.py path/to/evaluation_dataset.csv path/to/model_checkpoint.ckpt 

import argparse
import numpy as np
import pandas as pd
import torch
import os
import logging
from model import ConvLSTMCAMbiotic
from data_preprocessing1 import process_smiles, load_data, smiles_chars
from data_preprocessing2 import smiles_to_sequences, pad_one_hot_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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




def preprocess_and_filter_data(file_path, target_column='TARGET'):
    # Load data
    data = pd.read_csv(file_path)
    
    # Check for PROCESSED_SMILES column and choose the appropriate SMILES column
    if 'PROCESSED_SMILES' in data.columns:
        smiles_column = 'PROCESSED_SMILES'
        logger.info("Using pre-standardized SMILES from 'PROCESSED_SMILES' column.")
        # Skip standardization as the SMILES are already processed
        valid_data = data
    else:
        smiles_column = 'SMILES'
        logger.info("Standardizing SMILES from 'SMILES' column.")
        data = process_smiles(data, smiles_column)
        valid_data = data.copy()


    # Validate and filter SMILES strings based on criteria
    filtered_out_data = []
    valid_data = data.copy()
    for index, row in valid_data.iterrows():
        reasons = validate_smiles(row[smiles_column], maxlen, smiles_chars)
        if reasons:
            filtered_out_data.append((row[smiles_column], ', '.join(reasons)))
            valid_data.drop(index, inplace=True)

    if filtered_out_data:
        pd.DataFrame(filtered_out_data, columns=[smiles_column, 'Reason']).to_csv('Invalid_smiles_not_evaluated.csv', index=False)
        logger.info("Filtered SMILES saved to 'Invalid_smiles_not_evaluated.csv'.")

    if valid_data.empty:
        raise ValueError("No valid SMILES strings found in the dataset.")

    # Process dataset for evaluation
    processed_data, labels = process_dataset_for_evaluation(valid_data, smiles_column, target_column, maxlen, smiles_chars)

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
    with torch.no_grad():
        inputs = torch.tensor(processed_data).float().to(model.device)
        outputs_tuple = model(inputs)  # This returns a tuple
        logits = outputs_tuple[0]  # Extract the logits which are the first item of the tuple
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()  # Softmax on logits
        predictions = np.argmax(probabilities, axis=1)  # Predicted class
        return probabilities, predictions

    

# Define the mapping from numerical targets to categorical labels
target_map = {0: 'inactive', 1: 'slightly active', 2: 'active'}

# Define the agreement categorization function
def categorize_agreement(original, prediction):
    if original == prediction:
        return 'AGREE'
    elif original == 'active' and prediction == 'slightly active':
        return 'PARTIALLY_AGREE_DOWNGRADED'
    elif original == 'slightly active' and prediction == 'active':
        return 'PARTIALLY_AGREE_UPGRADED'
    else:
        return 'DISAGREE'

def generate_agreement_status_summary(merged_df):
    agreement_status_summary = merged_df['Agreement_Status'].value_counts()
    summary_lines = []
    for category, count in agreement_status_summary.items():
        if category != 'NO_TARGET':
            summary_lines.append(f"{category}: {count}")
    if 'NO_TARGET' in agreement_status_summary.index:
        no_target_count = agreement_status_summary['NO_TARGET']
        summary_lines.append(f"NO_TARGET: {no_target_count}")
    return "\n".join(summary_lines)



def main(dataset_path, checkpoint_path):
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Checkpoint Path: {checkpoint_path}")

    if not os.path.exists(dataset_path) or not os.path.exists(checkpoint_path):
        logger.error("File not found.")
        return

    test_df = pd.read_csv(dataset_path)

    model = load_model(checkpoint_path)
    processed_data, labels = preprocess_and_filter_data(dataset_path, 'TARGET')
    
    probabilities, predictions = evaluate(model, processed_data, labels)
    logger.info(f"Shape of probabilities array: {probabilities.shape}")

    results_df = pd.DataFrame({
        'Prob_Inactive': probabilities[:, 0],
        'Prob_Slightly_Active': probabilities[:, 1],
        'Prob_Active': probabilities[:, 2],
    }, index=test_df.index)
    
    results_df['Final_Prediction'] = predictions
    results_df['Final_Prediction_Status'] = [target_map[pred] for pred in predictions]

    merged_df = test_df.join(results_df)

    if 'TARGET' in merged_df.columns:
        merged_df['Original_Target_Status'] = merged_df['TARGET'].map(target_map)
        merged_df['Agreement_Status'] = merged_df.apply(
            lambda row: categorize_agreement(row['Original_Target_Status'], row['Final_Prediction_Status']), axis=1
        )
    else:
        merged_df['Agreement_Status'] = 'NO_TARGET'

    agreement_status_summary = generate_agreement_status_summary(merged_df)
    logger.info("Agreement Status Summary:\n%s", agreement_status_summary)

    results_file_path = 'evaluation_results.csv' # Save the results to a CSV file
    merged_df.to_csv(results_file_path, index=False)
    logger.info(f"Evaluation completed. Results saved to '{results_file_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a new dataset using the trained model.')
    parser.add_argument('dataset_path', type=str, help='The file path to the new dataset CSV file.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the trained model checkpoint.')
    args = parser.parse_args()
    main(args.dataset_path, args.checkpoint_path)


