# CAMbiotics

## Description

This repository contains the implementation of a deep learning model for binary classification of chemical compounds represented as SMILES (Simplified Molecular Input Line Entry System) strings. The model architecture combines convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to identify  bioactive compounds for antimicrobial activity (Include the microbial specie the model was trained for). Additionally, it provides tools for interpreting the model's predictions, such as generating Class Activation Maps (CAMs) to highlight important structural motifs.

## Installation

Before you can run the scripts, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch
- PyTorch Lightning
- RDKit
- Torchcam
- Matplotlib
- NumPy
- Pandas
- Scikit-learn

You can install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

Please note that RDKit might require a different installation approach depending on your operating system. Refer to the [official RDKit documentation](https://www.rdkit.org/docs/Install.html) for installation instructions.

## Usage

To train the model with your dataset, ensure that your CSV file is formatted correctly with the required columns: 'SMILES', 'TARGET', and 'COMPOUND_ID'. Then, run the following command:

```bash
python main.py path/to/your/dataset.csv
```

### Parameters

You can fine-tune the model by specifying the following parameters:

- `--vocab_size` : The size of the vocabulary (default is 100)
- `--embedding_dim` : The size of the embedding vectors (default is 128)
- `--cnn_filters` : The number of filters in the convolutional layers (default is 64)
- `--lstm_units` : The number of units in the LSTM layer (default is 64)
- `--output_size` : The number of units in the output layer (default is 1)
- `--learning_rate` : The learning rate for the optimizer (default is 1e-3)

### Example

```bash
python main.py data.csv --vocab_size 120 --embedding_dim 150 --lstm_units 80
```

## Files and Modules Description

- `data_preprocessing.py`: Contains functions for loading data, encoding SMILES, padding sequences, and splitting the dataset.
- `data_loader.py`: PyTorch Lightning data module for handling data loading.
- `model.py`: Defines the SMILESClassifier, a neural network for SMILES classification.
- `visualization.py`: Contains functions for generating and saving visualizations of the molecule with activation mappings.
- `main.py`: The main script to train the model using the specified dataset and hyperparameters.

## Contributing

Contributions to improve the model or any part of the project are welcome. Please feel free to fork the repository and submit pull requests.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact


Project Link: https://github.com/catenate15/CAMbiotics

## Acknowledgements



