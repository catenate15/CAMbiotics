import torch
import json
from torchcam.methods import GradCAM
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import matplotlib.cm as cm
from IPython.display import SVG, display
import matplotlib.colors as mcolors
from PIL import Image
import cairosvg

# Define a colormap that transitions from red to green through black. Do not confuse the blue in the dictionary with the blue in the colormap. The colomap is red-green-black, not red-green-blue.
cdict = {
    'red': ((0.0, 1.0, 1.0),  # Red at the lowest end
            (0.5, 0.0, 0.0),  # Transition to black in the middle
            (1.0, 0.0, 0.0)), # Stay black at the highest end (no red component)

    'green': ((0.0, 0.0, 0.0), # No green at the lowest end
              (0.5, 0.0, 0.0), # Transition to black in the middle
              (1.0, 1.0, 1.0)), # Green at the highest end

    'blue': ((0.0, 0.0, 0.0),  # No blue at the lowest end
             (0.5, 0.0, 0.0),  # No blue in the middle
             (1.0, 0.0, 0.0))  # No blue at the highest end
}


# Create the colormap
custom_cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)




# Load char_to_index mapping
def load_char_to_index(filepath='char_to_index.json'):
    with open(filepath, 'r') as f:
        char_to_index = json.load(f)
    return char_to_index

# Function to convert SMILES string to tensor using one-hot encoding
def smiles_to_tensor(smiles_string, char_to_index):
    tensor = torch.zeros(len(smiles_string), len(char_to_index))
    for i, char in enumerate(smiles_string):
        if char in char_to_index:
            tensor[i, char_to_index[char]] = 1
        else:
            print(f"Warning: Character {char} not in the SMILES alphabet")
    return tensor.unsqueeze(0)  # Add batch dimension


def map_activations_to_smiles(activation_map, smiles_string):
    # Convert the SMILES string to a molecule object
    mol = Chem.MolFromSmiles(smiles_string)

    # Initialize an empty dictionary to store the mapped activations
    mapped_activations = {}

    # Iterate over the atoms in the molecule
    for i, atom in enumerate(mol.GetAtoms()):
        # Use the atom's index as the key and the corresponding activation as the value
        mapped_activations[i] = activation_map[i]

    return mapped_activations

# Function to overlay activations on the molecule
def overlay_activations_on_molecule(molecule, mapped_activations):
    # Generate a 2D conformation of the molecule
    molecule.Compute2DCoords()

    # Prepare the highlight map
    highlight_map = {}
    for atom_idx, activation in mapped_activations.items():
        # Convert activation to color using the custom colormap
        color = custom_cmap(activation)
        highlight_map[atom_idx] = color

    # Draw the molecule with the highlight map
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    opts = drawer.drawOptions()
    # Set any RDKit drawing options here
    drawer.DrawMolecule(molecule, highlightAtoms=highlight_map.keys(), highlightAtomColors=highlight_map)
    drawer.FinishDrawing()

    # Get the SVG string of the molecule with highlights
    svg = drawer.GetDrawingText()
    return svg

# Function to generate activation map and visualize it
def generate_activation_map(model, smiles_string, char_to_index):
    # Convert SMILES to tensor
    input_tensor = smiles_to_tensor(smiles_string, char_to_index)

    # Get the class activation map
    cam_extractor = GradCAM(model)
    activation_map = cam_extractor(input_tensor)
    
    # Map activations back to SMILES characters/substructures
    mapped_activations = map_activations_to_smiles(activation_map, smiles_string)
    
    # Convert SMILES to a 2D molecule representation
    molecule = Chem.MolFromSmiles(smiles_string)
    molecule_with_highlights = overlay_activations_on_molecule(molecule, mapped_activations)
    
    # Create a color-coded visualization
    visualization = create_color_coded_visualization(molecule_with_highlights)
    
    return visualization





    # Function to save the visualization as an image file
def save_image(svg_string, file_path='visualization.png'):
    cairosvg.svg2png(bytestring=svg_string, write_to=file_path)
    print(f"Saved visualization as {file_path}")


def create_color_coded_visualization(molecule_svg):
    # Render the SVG in a Jupyter notebook
    display(SVG(molecule_svg))

    # Convert SVG data to an image
# Usage
char_to_index = load_char_to_index()  # Load the char_to_index mapping
smiles_string = "Your SMILES String Here" # Replace with your SMILES string
# Assuming `model` is your trained model and `smiles_to_tensor` is a function
model = None  # Replace with actual model loading

# Generate the activation map and visualize or save
molecule_svg = generate_activation_map(model, smiles_string, char_to_index)
create_color_coded_visualization(molecule_svg)  # For Jupyter notebook display
save_image(molecule_svg, 'molecule_visualization.png')  # To save as an image file
