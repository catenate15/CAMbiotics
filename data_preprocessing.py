import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from molvs import standardize_smiles

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
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler(f'{base_file_name}_datapreprocessing1.log')  # File handler

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

def standardize_smiles_column(data, smiles_column):
    if smiles_column not in data.columns:
        raise ValueError(f"Column '{smiles_column}' not found in the DataFrame.")
    data['STANDARDIZED_SMILES'] = data[smiles_column].apply(standardize_smiles)
    return data

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
    """
    Main function to run the preprocessing steps on the provided CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
    """
    base_file_name = os.path.splitext(os.path.basename(csv_file))[0]
    logger = setup_logger(base_file_name)
    logger.info("Starting data preprocessing...")
    try:
        data = load_data(csv_file, ['SMILES'])
        data = standardize_smiles_column(data, 'SMILES')
        data.to_csv(f'standardized_{os.path.basename(csv_file)}', index=False)
        filtered_data = filter_smiles_by_embedding(data, 'STANDARDIZED_SMILES', smiles_chars)
        get_maxlen_histogram(filtered_data, 'STANDARDIZED_SMILES', csv_file)
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing1.py <path_to_dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
