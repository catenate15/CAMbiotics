import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem

# Function to load the data
def load_data(file_path, augment=False, num_augmentations=10):
    try:
        data = pd.read_csv(file_path)
        
        # Check for required columns
        for col in ['SMILES', 'TARGET', 'COMPOUND_ID']:
            if col not in data.columns:
                raise ValueError(f"Column {col} is missing from the data.")
        
        smiles_list = data['SMILES'].tolist()
        activities_list = data['TARGET'].tolist()  # Get the activities as a list
        compound_ids_list = data['COMPOUND_ID'].tolist()  # Get the compound ids as a list
        
        if augment:
            augmented_smiles_list = []
            augmented_activities_list = []
            augmented_compound_ids_list = []
            for i, smiles in enumerate(smiles_list):
                augmented_smiles = augment_smiles(smiles, num_augmentations=num_augmentations)
                augmented_smiles_list.extend(augmented_smiles)
                # Replicate the activity label for each augmented SMILE
                augmented_activities_list.extend([activities_list[i]] * len(augmented_smiles))
                # Replicate the compound ID for each augmented SMILE
                augmented_compound_ids_list.extend([compound_ids_list[i]] * len(augmented_smiles))
            
            smiles_list = augmented_smiles_list
            activities_list = augmented_activities_list
            compound_ids_list = augmented_compound_ids_list

        return smiles_list, activities_list, compound_ids_list

    except Exception as e:
        print(f"Failed to load data: {e}")
        return [], [], [] # Return empty lists if there's an error


# Function to create a character to index mapping and encode SMILES
def smiles_to_seq(smiles_list):
    chars = set(''.join(smiles_list))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    smiles_enc = []

    # Convert SMILES to integer-encoded sequences and filter invalid ones
    invalid_smiles = []
    for smiles in smiles_list:
        try:
            smiles_code = [char_to_idx[char] for char in smiles]
            smiles_enc.append(smiles_code)
        except KeyError as e:
            invalid_smiles.append(smiles)

    if invalid_smiles:
        print(f"Invalid SMILES that couldn't be encoded: {invalid_smiles}")

    return smiles_enc, char_to_idx, invalid_smiles

# Function to augment a SMILES string
def augment_smiles(smiles, num_augmentations=10):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Invalid original SMILES string
        return []
    
    augmented_smiles = [smiles]  # Include the original SMILES string
    for _ in range(num_augmentations - 1):
        augmented_mol = Chem.Mol(mol)
        AllChem.RandomizeMol(augmented_mol)
        new_smiles = Chem.MolToSmiles(augmented_mol, canonical=False)
        if Chem.MolFromSmiles(new_smiles):  # Check if the new SMILES is valid
            augmented_smiles.append(new_smiles)
    
    return augmented_smiles


# check if the SMILES string is valid
def create_char_to_index(smiles_list):
    unique_chars = set(''.join(smiles_list))  # Get all unique characters from all SMILES
    return {char: idx for idx, char in enumerate(sorted(unique_chars))}



# Function to pad the sequences
def pad_smiles_sequences(smiles_enc):
    max_len = max(len(smiles) for smiles in smiles_enc)
    print(f"The maximum length of SMILES strings (max_len) is: {max_len}")
    padded_smiles = [seq + [0]*(max_len - len(seq)) for seq in smiles_enc]
    return np.array(padded_smiles)


# Function to check data imbalance
def check_data_balance(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    print("Data distribution before splitting:")
    for class_label, count in counter.items():
        print(f"Class {class_label}: {count/total:.2%} of the dataset")

# Function to split the data into train, validation, and test sets with balanced class distributions
def split_data(smiles, labels, train_size=0.7, valid_size=0.15, test_size=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(smiles, labels, train_size=train_size, stratify=labels)
    valid_test_split = valid_size / (valid_size + test_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_test_split, stratify=y_temp)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Function to save the processed data
def save_processed_data(X, y, base_file_path, data_type):
    # Save the processed data to CSV files with appropriate naming convention
    processed_data = pd.DataFrame({'Encoded_SMILES': list(X), 'Activity': y})
    save_path = f"{base_file_path}_{data_type}.csv"
    processed_data.to_csv(save_path, index=False)
    print(f"Processed {data_type} data saved to {save_path}")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the dataset.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing the dataset.')
    return parser.parse_args()

# Example usage
if __name__ == '__main__':
    # Assuming command-line argument parsing is enabled
    args = parse_arguments()

    # Load the dataset
    smiles, activities, compound_ids = load_data(args.file_path)

    # Convert SMILES to sequences
    smiles_enc, char_to_idx, invalid_smiles = smiles_to_seq(smiles)

    # Report any invalid SMILES strings
    if invalid_smiles:
        print("Invalid SMILES strings (excluded from dataset):")
        invalid_data = pd.DataFrame({'COMPOUND_ID': compound_ids, 'SMILES': smiles})
        print(invalid_data[invalid_data['SMILES'].isin(invalid_smiles)])

    # Pad sequences
    padded_smiles = pad_smiles_sequences(smiles_enc)

    # Split dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(padded_smiles, activities, train_size=0.7, valid_size=0.15, test_size=0.15)

    # Print class proportions in each set
    for dataset, y, name in [(X_train, y_train, "Train"), (X_valid, y_valid, "Validation"), (X_test, y_test, "Test")]:
        class_counts = pd.Series(y).value_counts(normalize=True)
        print(f"\nClass proportions in {name} set:")
        print(class_counts)

    # Save the processed data
    base_file_path = args.file_path.split('.')[0]  # Removes the file extension
    save_processed_data(X_train, y_train, base_file_path, 'Train')
    save_processed_data(X_valid, y_valid, base_file_path, 'Valid')
    save_processed_data(X_test, y_test, base_file_path, 'Test')

    # Print the first 5 encoded and padded SMILES for verification
    print("\nFirst 5 encoded and padded SMILES:")
    for i in range(5):
        print(padded_smiles[i])

    # Create and save the char_to_index mapping
    char_to_index = create_char_to_index(smiles)
    with open('char_to_index.json', 'w') as f:
        json.dump(char_to_index, f)

    # Print the first 5 entries of char_to_index for verification
    print("First 5 entries in char_to_index:")
    for idx, (char, index) in enumerate(char_to_index.items()):
        print(f"'{char}': {index}")
        if idx == 4:  # Stop after printing the first 5 entries
            break
