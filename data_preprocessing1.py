import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from molvs.fragment import FragmentRemover
from molvs.standardize import Standardizer
from molvs.normalize import Normalizer
from molvs.validate import Validator

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.propagate = False  # Prevents duplication of log messages in the outputs if false

# Define the SMILES characters that are allowed in the dataset
smiles_chars = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']
                

def setup_logger(base_file_name):
    """Sets up a logger to log to both console and file."""
    logger = logging.getLogger(__name__) # ensures that logger is named after the module
    logger.setLevel(logging.INFO)

    # Creates two handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(f'{base_file_name}_datapreprocessing.log')  # File handler

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

def load_data(file_path, required_columns):
    data = pd.read_csv(file_path)
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in the data: {missing_cols}")
    return data

def process_smiles(data, smiles_column):
    """
    Applies fragment removal and normalization to the SMILES column.
    """
    fragment_remover = FragmentRemover()
    normalizer = Normalizer()

    def process_smile(smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            mol = fragment_remover.remove(mol)
            mol = normalizer.normalize(mol)
            return MolToSmiles(mol)
        except Exception as e:
            logger.error(f"Error in processing SMILES '{smile}': {e}")
            return None

    data['PROCESSED_SMILES'] = data[smiles_column].apply(process_smile)
    return data

def filter_smiles_by_embedding(data, smiles_column, smiles_chars):
    valid_mask = data[smiles_column].apply(lambda smile: all(char in smiles_chars for char in smile))
    filtered_data = data[valid_mask]
    filtered_out_data = data[~valid_mask]
    filtered_out_count = len(filtered_out_data)
    filtered_percentage = (filtered_out_count / len(data)) * 100
    logger.info(f"Filtered out {filtered_out_count} SMILES ({filtered_percentage:.2f}%) not conforming to the predefined embedding.")
    if filtered_out_count > 0:
        filtered_out_csv = f"{os.path.splitext(file_path)[0]}_filtered_out.csv"
        filtered_out_data.to_csv(filtered_out_csv, index=False)
        logger.info(f"Filtered out data saved to {filtered_out_csv}")
    return filtered_data

def summarize_activity(data, activity_column):
    """
    Summary of the percentage of 'inactive', 'slightly active', and 'active'.
    """
    summary = data[activity_column].value_counts(normalize=True) * 100
    logger.info(f"Summary of activity categories:\n{summary}")

def get_maxlen_histogram(data, smiles_column, csv_file_path):
    lengths = data[smiles_column].apply(len)
    histogram_path = f"{os.path.splitext(csv_file_path)[0]}_smiles_lengths_histogram.png"
    plt.hist(lengths, bins=range(0, max(lengths) + 1, 1))
    plt.title("Distribution of SMILES Lengths")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.savefig(histogram_path)
    logger.info(f"Histogram of SMILES lengths saved to {histogram_path}")

def main(csv_file):
    base_file_name = os.path.splitext(os.path.basename(csv_file))[0]
    logger = setup_logger(base_file_name)
    logger.info("Starting data preprocessing...")
    try:
        data = load_data(csv_file, ['standardized_smiles', 'TARGET'])
        data = process_smiles(data, 'standardized_smiles')
        data = data[data['PROCESSED_SMILES'].notnull()]
        summarize_activity(data, 'TARGET')
        filtered_data = filter_smiles_by_embedding(data, 'PROCESSED_SMILES', smiles_chars)
        get_maxlen_histogram(filtered_data, 'PROCESSED_SMILES', csv_file)
        # Save the processed and filtered dataset
        processed_csv_path = f"{os.path.splitext(csv_file)[0]}_processed.csv"
        filtered_data.to_csv(processed_csv_path, index=False)
        logger.info(f"Processed dataset saved to {processed_csv_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
