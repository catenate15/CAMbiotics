import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import logging
import numpy as np
from itertools import permutations
import random
from rdkit.Chem import Descriptors
import sys
from data_preprocessing1 import smiles_chars

logger = logging.getLogger(__name__)
# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def calculate_molecular_weight(smiles):
    """ Calculate the molecular weight of a molecule given its SMILES string. """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)
    else:
        return None
    
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


def filter_smiles(data, smiles_column, maxlen, smiles_chars, base_file_path, target_column='TARGET', max_mol_wt=650):
    """
    ...
    Args:
        ...
        max_mol_wt (float): The maximum molecular weight allowed for a molecule.
        ...
    """
    # Calculate molecular weights
    data['MOL_WT'] = data[smiles_column].apply(calculate_molecular_weight)

    # Add reason for filtering if molecular weight exceeds max_mol_wt
    def filter_reason(smiles, mol_wt):
        reason = []
        if len(smiles) > maxlen:
            reason.append('maxlen exceeded')
        if not all(char in smiles_chars for char in smiles):
            reason.append('invalid character')
        if mol_wt is None or mol_wt > max_mol_wt:
            reason.append('mol wt exceeded')
        return ', '.join(reason) or 'valid'
    
    # Apply the filter_reason function to each SMILES string along with its molecular weight
    data['REASON'] = data.apply(lambda x: filter_reason(x[smiles_column], x['MOL_WT']), axis=1)

    # Split the data into filtered and filtered-out based on the reason
    filtered_data = data[data['REASON'] == 'valid']
    filtered_out_data = data[data['REASON'] != 'valid']

    # Calculate overall and class-specific filtered-out percentages
    total_filtered_out_count = len(filtered_out_data)
    total_count = len(data)
    overall_filtered_out_percent = (total_filtered_out_count / total_count) * 100

    # Adjustments to include slightly active compounds
    actives_filtered_out_count = filtered_out_data[filtered_out_data[target_column] == 2].shape[0]  # Assuming '2' is active
    slightly_actives_filtered_out_count = filtered_out_data[filtered_out_data[target_column] == 1].shape[0]  # Assuming '1' is slightly active
    inactives_filtered_out_count = filtered_out_data[filtered_out_data[target_column] == 0].shape[0]  # Assuming '0' is inactive

    actives_percent = (actives_filtered_out_count / total_count) * 100
    slightly_actives_percent = (slightly_actives_filtered_out_count / total_count) * 100
    inactives_percent = (inactives_filtered_out_count / total_count) * 100

    # Log the details including slightly active compounds
    logger.info(f"Filtered out {total_filtered_out_count} SMILES ({overall_filtered_out_percent:.2f}%) not conforming to the predefined embedding.")
    logger.info(f"  - Actives filtered out: {actives_filtered_out_count} ({actives_percent:.2f}%)")
    logger.info(f"  - Slightly Actives filtered out: {slightly_actives_filtered_out_count} ({slightly_actives_percent:.2f}%)")  # Added line
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
        logger.info(f"  Class {value} (0: inactive, 1: slightly active, 2: active): {count} counts, {class_proportions[value] * 100:.2f}%")



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
    # Mapping characters to integers
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, maxlen)

    # Pad the one-hot encoded sequences
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, maxlen)

    # Extract labels
    labels = data[target_column].values
    
    # Select only the necessary columns for the output
    data_subset = data[['COMPOUND_ID', smiles_column, target_column]]
    
    return padded_smiles, labels, data_subset



def split_and_save_data(train_data, valid_data, test_data, smiles_column, target_column, base_file_path, maxlen):
    try:
        # Remove the '.csv' extension from the base_file_path if present
        if base_file_path.endswith('.csv'):
            base_file_path = base_file_path[:-4]  # Remove the last 4 characters, '.csv'

        # Now, base_file_path doesn't include '.csv', and you can append new suffixes without issue
        columns_to_save = ['COMPOUND_ID', smiles_column, target_column]

        # Process and save training data
        train_features, train_labels, train_data_subset = process_dataset(train_data, smiles_column, target_column, maxlen)
        train_data_subset = train_data_subset[columns_to_save]
        np.save(f'{base_file_path}_train_features.npy', train_features)
        train_data_subset.to_csv(f'{base_file_path}_train_labels.csv', index=False)

        # Repeat for validation and test data
        valid_features, valid_labels, valid_data_subset = process_dataset(valid_data, smiles_column, target_column, maxlen)
        valid_data_subset = valid_data_subset[columns_to_save]
        np.save(f'{base_file_path}_valid_features.npy', valid_features)
        valid_data_subset.to_csv(f'{base_file_path}_valid_labels.csv', index=False)

        test_features, test_labels, test_data_subset = process_dataset(test_data, smiles_column, target_column, maxlen)
        test_data_subset = test_data_subset[columns_to_save]
        np.save(f'{base_file_path}_test_features.npy', test_features)
        test_data_subset.to_csv(f'{base_file_path}_test_labels.csv', index=False)

        logger.info("Data successfully processed and saved for training, validation, and test sets.")
    except Exception as e:
        logger.error(f"An error occurred while saving the files: {e}")
        return None  # Return None explicitly on error

    return {
        'train_features': f'{base_file_path}_train_features.npy',
        'train_labels': f'{base_file_path}_train_labels.csv',
        'valid_features': f'{base_file_path}_valid_features.npy',
        'valid_labels': f'{base_file_path}_valid_labels.csv',
        'test_features': f'{base_file_path}_test_features.npy',
        'test_labels': f'{base_file_path}_test_labels.csv'
    }



def preprocess_data(csv_file, smiles_column='SMILES', target_column='TARGET', augment=False, num_augmentations=10):
    standardized_file_path = csv_file

    # Load and validate data
    data = load_data(standardized_file_path, [smiles_column, target_column])

    logger.info("Initial class proportions:")
    log_class_proportions(data[target_column], "Initial")

    # Define and filter SMILES strings
    maxlen = 350
    filtered_data = filter_smiles(data, smiles_column, maxlen, smiles_chars, standardized_file_path, target_column, max_mol_wt=650)
    
    # Process and split data
    train_data, valid_data, test_data = preprocess_and_split_data(filtered_data, smiles_column, target_column)

    #  augment trainng data only
    if augment:
        train_data = augment_data(train_data, num_augmentations, smiles_column, target_column)

    # Save processed datasets
    processed_paths = split_and_save_data(train_data, valid_data, test_data, smiles_column, target_column, standardized_file_path, maxlen)

    return processed_paths


def main(csv_file):
    logger = setup_logger(os.path.splitext(os.path.basename(csv_file))[0])
    logger.info("Starting data preprocessing...")

    standardized_file = csv_file  # Direct use without altering the filename

    try:
        # Load and preprocess data
        processed_paths = preprocess_data(standardized_file, 'PROCESSED_SMILES', 'TARGET', augment=True, num_augmentations=10)
        logger.info(f"Data preprocessing completed. Files saved at: {processed_paths}")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing2.py <path_to_dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])

