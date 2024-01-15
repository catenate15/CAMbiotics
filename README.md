# CAMbiotics

## Description

This repository contains the implementation of a deep learning model for binary classification of chemical compounds represented as SMILES (Simplified Molecular Input Line Entry System) strings. The model architecture combines convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to identify  bioactive compounds for antimicrobial activity (Include the microbial specie the model was trained for). Additionally, it provides tools for interpreting the model's predictions, such as generating Class Activation Maps (CAMs) to highlight important structural motifs.

## Installation
To set up the project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/catenate15/CAMbiotics
   cd your-repository
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the project, follow these steps:

1. **Data Preprocessing**
   Place your dataset in CSV format in the project directory and run:
   ```bash
   python main.py path/to/your-dataset.csv
   ```

2. **Training the Model**
   The model training is initiated by the `main.py` script. Customize training parameters using command-line arguments.

3. **Visualization**
   To visualize the activation maps on SMILES strings:
   ```bash
   python visualization.py
   ```

## Project Structure
- `data_preprocessing.py`: Module for preprocessing SMILES data.
- `data_loader.py`: PyTorch data loading utility.
- `model.py`: Definition of the CNN + LSTM model.
- `visualization.py`: Visualization tools for activation maps.
- `main.py`: Entry point for training and evaluating the model.

## Modules Overview

### 1. Data Preprocessing (`data_preprocessing.py`)

This script is responsible for preparing the raw SMILES dataset for the machine learning pipeline. Key functionalities include:

- **SMILES Augmentation**: Generates multiple augmented versions of each SMILES string to enrich the dataset.
- **SMILES Encoding**: Converts SMILES strings into integer sequences based on a character-to-index mapping.
- **Data Splitting**: Splits the dataset into training, validation, and test sets.
- **File Generation**: Outputs processed data files and a JSON file for the character-to-index mapping.

Run this script separately on a CPU for efficient processing.

#### Usage

```bash
python data_preprocessing.py <path_to_dataset.csv> --augment --num_augmentations 10
```

### 2. Model Training and Evaluation (`main.py`)

This is the central script for training and evaluating the neural network model. It handles:

- **Model Initialization**: Sets up the SMILESClassifier with specified hyperparameters.
- **Data Loading**: Uses the preprocessed data for training and validation.
- **Training Process**: Orchestrates the training and evaluation cycles using PyTorch Lightning.

This script is designed for GPU execution to leverage accelerated computing resources.

#### Usage

```bash
python main.py --data_path <preprocessed_data_path> --epochs 10 --batch_size 64
```

### 3. Visualization (`visualization.py`)

After training, this script visualizes the activations of SMILES strings in the neural network. Key features include:

- **Activation Mapping**: Uses Grad-CAM to highlight which parts of the SMILES strings are most influential in the model's predictions.
- **Molecular Visualization**: Renders the SMILES strings as 2D molecular structures with overlaid activations.

Run this script separately, as visualization can be resource-intensive.

#### Usage

```bash
python visualization.py
```

## Recommendations

- Execute `data_preprocessing.py` first to prepare your dataset for training.
- Follow with `main.py` for model training and evaluation.
- Use `visualization.py` to generate insightful visualizations of the model's predictions.
- Ensure that each script's dependencies and environment settings are correctly configured before execution.

---

