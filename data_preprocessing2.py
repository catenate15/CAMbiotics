import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
import random

logger = logging.getLogger(__name__)
# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# define SMILES characters ----------------------------------------------------
smiles_chars = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']
print(f"The length of smiles character is {len(smiles_chars)}")

def setup_logger(base_file_name):
    """Sets up a logger to log to both console and file."""
    logger = logging.getLogger(__name__) # ensures that logger is named after the module
    logger.setLevel(logging.INFO)

    # Creates two handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler(f'{base_file_name}_datapreprocessing2.log')  # File handler

    # Create formatters and add it to handlers
    format = '%(asctime)s - %(levelname)s - %(message)s'
    c_format = logging.Formatter(format)
    f_format = logging.Formatter(format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def validate_columns(data, required_columns):
    """ Validate that the data contains all required columns """
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in the data: {missing_cols}")
    


def load_data(file_path, required_columns):
    """ Load data from a CSV file and validate that it contains all required columns """
    data = pd.read_csv(file_path)
    validate_columns(data, required_columns)
    print(f"Data loaded from {file_path}:\n{data.head(2)}")
    return data


def filter_smiles(data, smiles_column, maxlen, smiles_chars, base_file_path, target_column = 'TARGET'):
    """
    Filter out SMILES strings that are too long or contain invalid characters, 
    and save the filtered-out SMILES with reasons. Additionally, report the percentage
    of actives and inactives filtered out.
    Args:
        data (DataFrame): The dataset containing SMILES strings.
        smiles_column (str): The column name for SMILES strings.
        maxlen (int): The maximum length of SMILES strings.
        smiles_chars (set): Set of valid characters in SMILES strings.
        base_file_path (str): Base file name to save the filtered-out SMILES.
        target_column (str): The column name for the target variable.
    Returns:
        DataFrame: The filtered dataset.
    """
    def filter_reason(smiles):
        reason = []
        if len(smiles) > maxlen:
            reason.append('maxlen exceeded')
        if not all(char in smiles_chars for char in smiles):
            reason.append('invalid character')
        return ', '.join(reason) or 'valid'

    # Apply the filter_reason function to each SMILES string
    data['REASON'] = data[smiles_column].apply(filter_reason)

    # Split the data into filtered and filtered-out based on the reason
    filtered_data = data[data['REASON'] == 'valid']
    filtered_out_data = data[data['REASON'] != 'valid']

    # Calculate overall and class-specific filtered-out percentages
    total_filtered_out_count = len(filtered_out_data)
    total_count = len(data)
    overall_filtered_out_percent = (total_filtered_out_count / total_count) * 100

    actives_filtered_out_count = filtered_out_data[filtered_out_data[target_column] == 1].shape[0]
    inactives_filtered_out_count = filtered_out_data[filtered_out_data[target_column] == 0].shape[0]
    actives_percent = (actives_filtered_out_count / total_count) * 100
    inactives_percent = (inactives_filtered_out_count / total_count) * 100

    # Log the details
    logger.info(f"Filtered out {total_filtered_out_count} SMILES ({overall_filtered_out_percent:.2f}%) not conforming to the predefined embedding.")
    logger.info(f"  - Actives filtered out: {actives_filtered_out_count} ({actives_percent:.2f}%)")
    logger.info(f"  - Inactives filtered out: {inactives_filtered_out_count} ({inactives_percent:.2f}%)")
    logger.info(f"Details saved in {base_file_path}_filtered_out.csv")

    # Save the filtered-out data
    filtered_out_csv = f"{base_file_path}_filtered_out.csv"
    filtered_out_data.to_csv(filtered_out_csv, index=False)

    return filtered_data

def pad_one_hot_sequences(sequences, maxlen):
    """ Pad one-hot encoded sequences to a maximum length"""
    one_hot_length = sequences.shape[2]
    padded_sequences = np.zeros((len(sequences), maxlen, one_hot_length), dtype=np.int8)

    for idx, sequence in enumerate(sequences):
        length = min(sequence.shape[0], maxlen)
        padded_sequences[idx, :length, :] = sequence[:length]

    return padded_sequences

def smile_to_sequence(smile, char_to_index, maxlen):
    encoded = np.zeros((maxlen, len(char_to_index)), dtype=np.int8)
    for i, char in enumerate(smile):
        if i < maxlen:  # Ensure index doesn't exceed maxlen - 1
            if char in char_to_index:
                encoded[i, char_to_index[char]] = 1
        else:
            break  # Break the loop if index equals or exceeds maxlen
    return encoded



def smiles_to_sequences(smiles_list, char_to_index, maxlen):
    sequences = []
    # Get 5 random indices from the smiles_list
    random_indices = random.sample(range(len(smiles_list)), 5)
    for i, smile in enumerate(smiles_list):
        original_length = len(smile)
        seq = smile_to_sequence(smile, char_to_index, maxlen)
        sequences.append(seq)
        if i in random_indices:  # Only print details for randomly selected sequences
            logger.info(f"Processed SMILE (random sample): {smile}, Original length: {original_length}, Sequence shape after padding: {seq.shape}")
    sequences_array = np.array(sequences)
    logger.info(f"Sequences array shape: {sequences_array.shape}")
    return sequences_array

def smiles_augmenter(smiles, num_generator=10, shuffle_limit=1000):
    """
    Generate augmented SMILES strings by permuting the atoms of a molecule.

    Args:
        smiles (str): The original SMILES string to augment.
        num_generator (int): The number of augmented SMILES strings to generate.
        shuffle_limit (int): The maximum number of attempts to generate unique SMILES.

    Returns:
        list: A list of unique augmented SMILES strings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    smiles_set = set()  # Using a set to avoid duplicates

    # For molecules with a small number of atoms, consider all permutations
    if num_atoms < 4:
        perms = list(permutations(range(num_atoms)))
        for p in perms:
            permuted_smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, p), canonical=False, isomericSmiles=True)
            smiles_set.add(permuted_smiles)

    else:
        # For larger molecules, randomly permute atoms
        count = 0
        while len(smiles_set) < num_generator and count < shuffle_limit:
            p = np.random.permutation(range(num_atoms))
            new_smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, p.tolist()), canonical=False, isomericSmiles=True)
            smiles_set.add(new_smiles)  # Add new SMILES if not in the set already
            count += 1

    return list(smiles_set)

def augment_data(data, num_augmentations, smiles_column, target_column):
    augmented_smiles = []
    augmented_labels = []
    original_ids = []
    original_smiles = []

    for _, row in data.iterrows():
        # Original data
        compound_id = row.get('COMPOUND_ID', 'Unknown')  # Assuming there's a COMPOUND_ID column
        orig_smile = row[smiles_column]
        label = row[target_column]

        # Generate augmented SMILES
        aug_smiles = smiles_augmenter(orig_smile, num_generator=num_augmentations)

        # Ensure the number of augmentations is consistent
        aug_smiles = aug_smiles[:num_augmentations]
        num_to_add = num_augmentations - len(aug_smiles)
        aug_smiles.extend([orig_smile] * num_to_add)

        # Extend lists
        augmented_smiles.extend([orig_smile] + aug_smiles)
        augmented_labels.extend([label] * (len(aug_smiles) + 1))
        original_ids.extend([compound_id] * (len(aug_smiles) + 1))
        original_smiles.extend([orig_smile] * (len(aug_smiles) + 1))

    #augment data linked back to original SMILES
    augmented_data = pd.DataFrame({
        'COMPOUND_ID': original_ids,
        'Original_SMILES': original_smiles,
        smiles_column: augmented_smiles,
        target_column: augmented_labels
    })

    return augmented_data




def log_class_proportions(y, dataset_name):
    class_counts = y.value_counts()
    total_count = len(y)
    class_proportions = class_counts / total_count
    logger.info(f"{dataset_name} set class proportions:")
    for value, count in class_counts.items():
        logger.info(f"  Class {value}: {count} counts, {class_proportions[value] * 100:.2f}%")



def preprocess_and_split_data(data, smiles_column, target_column, train_size=0.7, valid_size=0.15, test_size=0.15):
    # Split original data into training, validation, and test sets
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42, stratify=data[target_column])
    valid_data, test_data = train_test_split(temp_data, test_size=(test_size / (valid_size + test_size)), random_state=42, stratify=temp_data[target_column])

    # Printing class proportions in each set
    for dataset, name in zip([train_data, valid_data, test_data], ['Training', 'Validation', 'Test']):
        class_counts = dataset[target_column].value_counts()
        total_count = len(dataset)
        logger.info(f"{name} set class proportions:")
        for value, count in class_counts.items():
            logger.info(f"  Class {value}: {count} counts, {count / total_count * 100:.2f}% of {name} set")

    return train_data, valid_data, test_data

def process_dataset(data, smiles_column, target_column, maxlen):
    """
    Process the dataset to convert SMILES to one-hot encoding and pad them.
    """
    # Directly use smiles_chars for char_to_index mapping
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, maxlen)

    # Pad the one-hot encoded sequences
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, maxlen)

    # Extract labels
    labels = data[target_column].values

    return padded_smiles, labels, data



def split_and_save_data(train_data, valid_data, test_data, smiles_column, target_column, base_file_path, maxlen):
    try:
        # Process and save training data
        train_features, train_labels, _ = process_dataset(train_data, smiles_column, target_column, maxlen)
        np.save(f'{base_file_path}_train_features.npy', train_features)
        train_data[['COMPOUND_ID', 'SMILES', target_column]].to_csv(f'{base_file_path}_train_labels.csv', index=False)

        # Process and save validation data
        valid_features, valid_labels, _ = process_dataset(valid_data, smiles_column, target_column, maxlen)
        np.save(f'{base_file_path}_valid_features.npy', valid_features)
        valid_data[['COMPOUND_ID', 'SMILES', target_column]].to_csv(f'{base_file_path}_valid_labels.csv', index=False)

        # Process and save test data
        test_features, test_labels, _ = process_dataset(test_data, smiles_column, target_column, maxlen)
        np.save(f'{base_file_path}_test_features.npy', test_features)
        test_data[['COMPOUND_ID', 'SMILES', target_column]].to_csv(f'{base_file_path}_test_labels.csv', index=False)

        logger.info("Data successfully processed and saved for training, validation, and test sets.")
    except Exception as e:
        logger.error(f"An error occurred while saving the files: {e}")

    return {
        'train_features': f'{base_file_path}_train_features.npy',
        'train_labels': f'{base_file_path}_train_labels.csv',
        'valid_features': f'{base_file_path}_valid_features.npy',
        'valid_labels': f'{base_file_path}_valid_labels.csv',
        'test_features': f'{base_file_path}_test_features.npy',
        'test_labels': f'{base_file_path}_test_labels.csv'
    }




def preprocess_data(csv_file, smiles_column='SMILES', target_column='TARGET', augment=False, num_augmentations=10):
    # Extracting base file name
    base_file_name = os.path.basename(csv_file).rsplit('.', 1)[0]

    # Check if 'STANDARDIZED_' prefix is already included
    if not base_file_name.startswith("STANDARDIZED_"):
        base_file_name = f"STANDARDIZED_{base_file_name}"

    standardized_file_path = base_file_name


    # Check if processed files already exist
    processed_files_exist = all(
        os.path.exists(f"{standardized_file_path}_{suffix}.npy") or os.path.exists(f"{standardized_file_path}_{suffix}.csv")
        for suffix in ['train_features', 'valid_features', 'test_features', 'train_labels', 'valid_labels', 'test_labels']
    )
    if processed_files_exist:
        logger.info("Processed .npy and .csv files already exist. Skipping preprocessing.")
        return {suffix: f"{standardized_file_path}_{suffix}" for suffix in ['train_features', 'valid_features', 'test_features', 'train_labels', 'valid_labels', 'test_labels']}

    # Load and process data
    data = load_data(f"{standardized_file_path}.csv", [smiles_column, target_column])
    logger.info("Initial class proportions:")
    log_class_proportions(data[target_column], "Initial")

    # Define maxlen directly
    maxlen = 350  # this value is selected from what you chose from the histogram generated from data_preprocessing1.py

    # Filter out SMILES strings based on maxlen and smiles_chars
    filtered_data = filter_smiles(data, smiles_column, maxlen, smiles_chars, standardized_file_path)

    # Log details
    logger.info(f"Model will be trained on maxlen = {maxlen}. SMILES exceeding this length will not be used.")
    logger.info(f"Embedding used in model: {smiles_chars}")

    # Directly use smiles_chars for char_to_index mapping
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, maxlen)

    # Pad the one-hot encoded sequences
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, maxlen)

    # Log details for random samples
    random_smiles = random.sample(data[smiles_column].tolist(), 5)
    for smile in random_smiles:
        logger.info(f"Random SMILE: {smile}, Length after padding: {len(padded_smiles[data[smiles_column].tolist().index(smile)])}")

    # Extract labels
    labels = data[target_column].values

    # Split the filtered data into train, validation, and test sets
    train_data, valid_data, test_data = preprocess_and_split_data(filtered_data, smiles_column, target_column)

    # Augment only the training data
    if augment:
        train_data = augment_data(train_data, num_augmentations, smiles_column, target_column)
        logger.info(f"Training Data augmented to {len(train_data)} records")


    # Process and save each dataset
    print("Processing and saving split datasets...")
    processed_paths = split_and_save_data(train_data, valid_data, test_data, smiles_column, target_column, standardized_file_path, maxlen)

    return processed_paths


def main(csv_file):
    """
    Main function to run the preprocessing steps on the provided CSV file.
    Args:
        csv_file (str): Path to the CSV file containing the dataset.
    """
    base_file_name = os.path.splitext(os.path.basename(csv_file))[0]
    logger = setup_logger(base_file_name)
    logger.info("Starting data preprocessing...")

    try:
        # Constructing standardized dataset filename
        if base_file_name.startswith("STANDARDIZED_"):
            standardized_file = csv_file
        else:
            standardized_file = f"{base_file_name}.csv"

        # Check if the standardized dataset exists
        if not os.path.exists(standardized_file):
            logger.error(f"Standardized dataset file not found: {standardized_file}")
            sys.exit(1)

        # Run the preprocessing function with the path to the standardized CSV file
        processed_paths = preprocess_data(standardized_file, augment=True, num_augmentations=10)  # Set to False for no augmentation
        print(f"Data preprocessing completed. Files saved at: {processed_paths}")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import sys

    # Check if the script is run with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing2.py <path_to_dataset.csv>")
        sys.exit(1)

    # Call the main function with the CSV file path
    main(sys.argv[1])

