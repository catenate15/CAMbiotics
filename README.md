# CAMbiotics

## Description

This repository contains the implementation of a deep learning model for binary classification of chemical compounds represented as SMILES (Simplified Molecular Input Line Entry System) strings. The model architecture combines convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to identify  bioactive compounds for antimicrobial activity (Include the microbial specie the model was trained for). Additionally, it provides tools for interpreting the model's predictions, such as generating Class Activation Maps (CAMs) to highlight important structural motifs.


## Modules Description

- **Data Preprocessing (`Data_processing1.py` and `Data_processing2.py`)**: Scripts for cleaning SMILES data, including filtering, standardizing, and dataset partitioning.
- **Data Loader (`data_loader.py`)**: Implements a PyTorch Lightning DataModule for efficient data handling and batching.
- **Model (`model.py`)**: Defines the neural network architecture combining convolutional, LSTM layers, and multi-head attention, focusing on learning from the structural patterns in SMILES data.
- **Training (`trainer.py`)**: Facilitates model training with checkpoints, hyperparameter configuration, and TensorBoard logging.
- **Visualization (`visualization.ipynb`)**: Jupyter Notebook for visualizing activation maps on molecular structures, offering insights into the model's focal points.

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

### Data Preprocessing

1. **Initial Preprocessing**:
   - Run `Data_processing1.py` to clean and standardize SMILES data.
     ```sh
     python Data_processing1.py <path_to_raw_smiles.csv>
     ```

2. **Further Data Preparation**:
   - Execute `Data_processing2.py` for additional preprocessing, including filtering based on structural features and dataset splitting.
     ```sh
     python Data_processing2.py <path_to_standardized_smiles.csv>
     ```

### Model Training

- Train the model using `trainer.py`, specifying the path to the preprocessed dataset and desired hyperparameters.
  ```sh
  python trainer.py --data_path <path_to_preprocessed_data.csv> --epochs 100 --batch_size 64
  ```

### Visualization of Activation Maps

- To visualize the model's focus through activation maps on molecular structures, run the `visualization.ipynb` notebook in Jupyter:
  ```sh
  jupyter notebook visualization.ipynb
  ```
- Follow the instructions within the notebook to load a trained model checkpoint and visualize activation maps on specified SMILES strings.



## Contributing

Contributions are welcome. Please adhere to the project's coding conventions and commit guidelines. Open a pull request with detailed descriptions of your changes or enhancements.

---

