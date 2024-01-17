"""The sequential workflow of the script for processing input SMILES strings is as follows:

1. **Load and Validate Data**: The script starts by loading the CSV file containing the SMILES strings and bioactivity data. It checks for the presence of required columns and ensures that the data format is correct.

2. **Extract Unique Characters**: It identifies and saves the unique characters present in the SMILES strings, which are essential for the one-hot encoding process.

3. **Determine Maximum SMILES Length**: The script calculates the maximum length of the SMILES strings that the model will consider, based on a percentile threshold. This step includes generating and saving a histogram of the SMILES lengths distribution.

4. **Filter SMILES Strings**: The workflow includes a filtering step to remove SMILES strings that exceed the maximum length or contain characters not found in the set of unique characters extracted earlier.

5. **Data Augmentation ** :Tthe script generates new SMILES strings by permuting the atoms in the molecules, which can help the model to generalize better by training on a more diverse set of data.

6. **One-Hot Encode SMILES**: Each SMILES string is converted into a one-hot encoded matrix, where each character is represented as a binary vector of the presence or absence of characters from the unique set.

7. **Padding**: The one-hot encoded sequences are padded to ensure that they all have the same length, matching the maximum SMILES length determined earlier.

8. **Split Data**: The script splits the dataset into training, validation, and test sets, preserving the class balance across these sets to ensure a fair distribution of data for model training and evaluation.

9. **Save Processed Data**: Finally, all processed data, including the one-hot encoded matrices and the split datasets, are saved to files for easy access during the machine learning model development phase.

This workflow efficiently converts the raw SMILES strings into a structured format suitable for developing a predictive model for compound bioactivity. """

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


def get_unique_chars(data, smiles_column, output_path):
    """Extract the unique characters from SMILES strings and save them to a JSON file."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    if smiles_column not in data.columns:
        raise ValueError(f"'{smiles_column}' column not found in the DataFrame")

    unique_chars = set(''.join(data[smiles_column]))
    unique_chars = sorted(list(unique_chars))
    print(f"Unique characters: {unique_chars}")

    try:
        with open(output_path, 'w') as file:
            json.dump(unique_chars, file)
        logger.info(f"Unique characters in SMILES saved to {output_path}")
    except IOError as e:
        logger.error(f"Failed to save unique characters to {output_path}: {e}")
        raise

    return unique_chars





def get_maxlen(data, smiles_column, csv_file_path, percentile=0.95, bin_size=1):
    """Calculates the maximum length of SMILES strings, saves it to a text file, and generates a histogram.

    Args:
        data (pd.DataFrame): DataFrame containing the SMILES strings.
        smiles_column (str): Name of the column containing the SMILES strings.
        output_path (str): Path to save the output files.
        percentile (float, optional): Percentile to use for calculating the maximum length. Defaults to 0.95.
        bin_size (int, optional): Bin size for the histogram. Defaults to 1.

    Returns:
        int: The calculated maximum length.

    Raises:
        ValueError: If input data is not a pandas DataFrame or the specified column is not found.
        IOError: If an error occurs while saving the maximum length to the text file.
        Exception: If an error occurs while generating or saving the histogram.
    """
    print("Executing get_maxlen")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    if smiles_column not in data.columns:
        raise ValueError(f"'{smiles_column}' column not found in the DataFrame")

    lengths = data[smiles_column].apply(len)
    maxlen = int(lengths.quantile(percentile))
    print(f"Calculated maxlen: {maxlen}")

    # Extract base file name from csv_file_path
    base_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Constructing the file names for maxlen and histogram
    maxlen_file = f"{base_file_name}_maxlen.txt"
    histogram_path = f"{base_file_name}_smiles_lengths_histogram.png"

    # Saving the maxlen value
    try:
        with open(maxlen_file, 'w') as file:
            file.write(str(maxlen))
        logger.info(f"Maximum length of SMILES strings saved to {maxlen_file}")
    except IOError as e:
        logger.error(f"Failed to save maximum length to text file: {e}")
        raise

    # Saving the histogram
    try:
        plt.hist(lengths, bins=range(0, max(lengths) + bin_size, bin_size))
        plt.title("Distribution of SMILES Lengths")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.savefig(histogram_path)
        logger.info(f"Histogram of SMILES lengths saved to {histogram_path}")
    except Exception as e:
        logger.error(f"Failed to save histogram: {e}")
        raise

    return maxlen





def smile_to_sequence(smile, char_to_index, maxlen):
    """ Convert a SMILES string to a one-hot encoded array."""
    encoded = np.zeros((maxlen, len(char_to_index)), dtype=np.int8)
    for i, char in enumerate(smile):
        if i < maxlen and char in char_to_index:
            encoded[i, char_to_index[char]] = 1
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
            print(f"Processed SMILE (random sample): {smile}, Original length: {original_length}, Sequence shape after padding: {seq.shape}")
    sequences_array = np.array(sequences)
    print(f"Sequences array shape: {sequences_array.shape}")
    return sequences_array



def filter_smiles(data, smiles_column, maxlen, unique_chars, base_file_path):
    """
    Filter out SMILES strings that are too long or contain invalid characters, and save the filtered-out SMILES.
    Args:
        data (DataFrame): The dataset containing SMILES strings.
        smiles_column (str): The column name for SMILES strings.
        maxlen (int): The maximum length of SMILES strings.
        unique_chars (set): Set of valid characters in SMILES strings.
        base_file_name (str): Base file name to save the filtered-out SMILES.
    Returns:
        DataFrame: The filtered dataset.
    """
    def is_valid_smiles(smiles):
        if len(smiles) > maxlen:
            return False
        for char in smiles:
            if char not in unique_chars:
                return False
        return True

    # Create a mask for valid SMILES strings
    valid_mask = data[smiles_column].apply(is_valid_smiles)
    # Filtered data is where the mask is True
    filtered_data = data[valid_mask]
    # Filtered-out data is where the mask is False
    filtered_out_data = data[~valid_mask]


    # Save the filtered-out
    filtered_out_csv = f"{base_file_path}_filtered_out.csv"
    filtered_out_data.to_csv(filtered_out_csv, index=False)

    print(f"Filtered data:\n{filtered_data.head(2)}")
    print(f"Number of SMILES strings before filtering: {len(data)}")
    print(f"Number of SMILES strings after filtering: {len(filtered_data)}")
    print(f"Filtered out data saved to: {filtered_out_csv}")
    return filtered_data




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

    for _, row in data.iterrows():
        original_smiles = row[smiles_column]
        label = row[target_column]
        aug_smiles = smiles_augmenter(original_smiles, num_generator=num_augmentations)

        # Ensure the number of augmentations is consistent
        aug_smiles = aug_smiles[:num_augmentations]
        num_to_add = num_augmentations - len(aug_smiles)
        aug_smiles.extend([original_smiles] * num_to_add)

        augmented_smiles.extend([original_smiles] + aug_smiles)
        augmented_labels.extend([label] * (len(aug_smiles) + 1))

    augmented_data = pd.DataFrame({
        smiles_column: augmented_smiles,
        target_column: augmented_labels
    })

    return augmented_data



def pad_one_hot_sequences(sequences, maxlen):
    """ Pad one-hot encoded sequences to a maximum length"""
    one_hot_length = sequences.shape[2]
    padded_sequences = np.zeros((len(sequences), maxlen, one_hot_length), dtype=np.int8)

    for idx, sequence in enumerate(sequences):
        length = min(sequence.shape[0], maxlen)
        padded_sequences[idx, :length, :] = sequence[:length]

    return padded_sequences



def log_class_proportions(y, dataset_name):
    class_counts = y.value_counts()
    total_count = len(y)
    class_proportions = class_counts / total_count
    logger.info(f"{dataset_name} set class proportions:")
    for value, count in class_counts.items():
        logger.info(f"  Class {value}: {count} counts, {class_proportions[value] * 100:.2f}%")

def save_class_proportions(proportions, file_path):
    """ Save class proportions to a CSV file """
    proportions_df = pd.DataFrame(proportions)
    proportions_df.to_csv(file_path, index=False)
    logger.info(f"Class proportions saved to {file_path}")
    print(f" Class proportions saved to {file_path}")

def log_and_save_class_proportions(labels, base_file_path):
    class_counts = labels.value_counts()
    total_count = len(labels)
    class_proportions = class_counts / total_count
    logger.info("Dataset class proportions:")
    for value, count in class_counts.items():
        logger.info(f"  Class {value}: {count} counts, {class_proportions[value] * 100:.2f}%")

    # Save class proportions to a CSV file
    proportions_df = pd.DataFrame({'Class': class_counts.index, 'Proportion': class_proportions.values})
    proportions_file = f"{base_file_path}_class_proportions.csv"
    proportions_df.to_csv(proportions_file, index=False)
    logger.info(f"Class proportions saved to {proportions_file}")



def split_and_save_data(features, labels, base_file_path, train_size=0.7, valid_size=0.15, test_size=0.15):
    # Ensure the sum of sizes equals 1
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("Sum of train, validation, and test sizes must equal 1")

    # Split data into training and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, train_size=train_size, random_state=42, stratify=labels
    )
    
    # Calculate the relative sizes of validation and test sets from the remaining dataset
    temp_size = 1.0 - train_size
    relative_valid_size = valid_size / temp_size
    
    # Split temp into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_valid_size, random_state=42, stratify=y_temp
    )

    # Save the datasets to .npy and .csv files
    np.save(f'{base_file_path}_train_features.npy', X_train)
    pd.DataFrame({'Activity': y_train}).to_csv(f'{base_file_path}_train_labels.csv', index=False)
    

    np.save(f'{base_file_path}_valid_features.npy', X_valid)
    pd.DataFrame({'Activity': y_valid}).to_csv(f'{base_file_path}_valid_labels.csv', index=False)

    np.save(f'{base_file_path}_test_features.npy', X_test)
    pd.DataFrame({'Activity' : y_test}).to_csv(f'{base_file_path}_test_labels.csv', index=False)

    logger.info(f"Data split and saved to {base_file_path}_train_features.npy, {base_file_path}_valid_features.npy, {base_file_path}_test_features.npy and corresponding label .csv files")

    return {
        'train_features': f'{base_file_path}_train_features.npy',
        'valid_features': f'{base_file_path}_valid_features.npy',
        'test_features': f'{base_file_path}_test_features.npy',
        'train_labels': f'{base_file_path}_train_labels.csv',
        'valid_labels': f'{base_file_path}_valid_labels.csv',
        'test_labels': f'{base_file_path}_test_labels.csv'
    }





def preprocess_data(file_path, smiles_column='SMILES', target_column='TARGET', augment=False, num_augmentations=10):
    
    """ Preprocess data and save the processed files. """
    base_file_path, _ = os.path.splitext(file_path)
     # Check if the processed files already exist
    processed_files_exist = all(
        os.path.exists(f"{base_file_path}_{suffix}.npy") or os.path.exists(f"{base_file_path}_{suffix}.csv")
        for suffix in ['train_features', 'valid_features', 'test_features', 
                       'train_labels', 'valid_labels', 'test_labels']
    )

    if processed_files_exist:
        logger.info("Processed .npy and .csv files already exist. Skipping preprocessing.")
        return {
            'train_features': f'{base_file_path}_train_features.npy',
            'valid_features': f'{base_file_path}_valid_features.npy',
            'test_features': f'{base_file_path}_test_features.npy',
            'train_labels': f'{base_file_path}_train_labels.csv',
            'valid_labels': f'{base_file_path}_valid_labels.csv',
            'test_labels': f'{base_file_path}_test_labels.csv'
        }


    # Load data
    data = load_data(file_path, [smiles_column, target_column])
    logger.info("Initial class proportions:")
    log_class_proportions(data[target_column], "Initial")
    logger.info(f"Data loaded with {len(data)} records")

    # Determine unique characters
    unique_chars_path = f"{base_file_path}_unique_chars.json"
    unique_chars = get_unique_chars(data, smiles_column, unique_chars_path)

    # Call get_maxlen to ensure the creation of maxlen file
    maxlen = get_maxlen(data, smiles_column, file_path)


    # Now, you can safely load maxlen from the text file
    maxlen_file = f"{base_file_path}_maxlen.txt"
    with open(maxlen_file, 'r') as file:
        maxlen = int(file.read().strip())
        print(f"Reading maxlen from file: {maxlen_file}")

    # Log details
    logger.info(f"Model trained on maxlen = {maxlen}. SMILES exceeding this length will not be used.")
    logger.info(f"Unique characters used in model: {unique_chars}")
    
    # Filter out SMILES strings based on maxlen and unique characters
    print("Before filtering: ", len(data))
    filtered_data = filter_smiles(data, smiles_column, maxlen, unique_chars, base_file_path)
    print("After filtering: ", len(filtered_data))
    logger.info(f"Data filtered to {len(filtered_data)} records")


    # Check lengths after filtering
    if len(data) != len(data[target_column]):
        logger.error("Inconsistent lengths after filtering")
        raise ValueError("Inconsistent lengths after filtering")
    
    # Augment the data (if specified)
    if augment:
        print("Before augmentation: ", len(data))
        data = augment_data(data, num_augmentations, smiles_column, target_column)
        print("After augmentation: ", len(data))
        logger.info(f"Data augmented to {len(data)} records")

    # Check lengths after augmentation
    if len(data) != len(data[target_column]):
        logger.error("Inconsistent lengths after augmentation")
        raise ValueError("Inconsistent lengths after augmentation")

    # Extract labels after filtering and augmentation
    labels = data[target_column]
    print("Labels extracted: ", len(labels))

    # Load the unique characters for char_to_index mapping
    with open(unique_chars_path, 'r') as file:
        smiles_chars = json.load(file)
    char_to_index = {c: i for i, c in enumerate(smiles_chars)}

    # Convert SMILES to one-hot encoding
    print("Converting SMILES to one-hot encoding...")
    one_hot_encoded = smiles_to_sequences(data[smiles_column].tolist(), char_to_index, maxlen)

    # Pad the one-hot encoded sequences
    print("Padding one-hot encoded sequences...")
    padded_smiles = pad_one_hot_sequences(one_hot_encoded, maxlen)

    # Extract labels
    labels = data[target_column].values

    # Split the data and save the split datasets
    print("Splitting data into train, validation, and test sets...")
    saved_paths = split_and_save_data(padded_smiles, labels, base_file_path)

    # Log and save class proportions
    for split_name, split_path in saved_paths.items():
        if 'labels' in split_path:
            split_labels = pd.read_csv(split_path)['Activity']
            logger.info(f"Class proportions in {split_name} set:")
            log_class_proportions(split_labels, split_name.capitalize())
    
    #   Output the data types and shapes
    print(f"Data types - SMILES: {type(one_hot_encoded)}, Labels: {type(labels)}")
    print(f"SMILES array shape: {one_hot_encoded.shape}, Labels series length: {len(labels)}")
    
    return saved_paths  # Return the paths to the processed files

    






def main(csv_file):
    """
    Main function to run the preprocessing steps on the provided CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
    """
    try:
        # Run the preprocessing function with the path to the CSV file
        processed_paths = preprocess_data(csv_file, augment=True, num_augmentations=10) # set to False for no augmentation
        print(f"Data preprocessing completed. Files saved at: {processed_paths}")
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")

if __name__ == '__main__':
    import sys

    # Check if the script is run with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing.py <path_to_dataset.csv>")
        sys.exit(1)  # Exit the script indicating that there was an error in the input

    # Call the main function with the CSV file path
    main(sys.argv[1])


