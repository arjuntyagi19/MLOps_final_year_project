# app.py

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# RDKit for molecule handling
from rdkit import Chem
# from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# For generating images in Streamlit
from PIL import Image
import io

# For handling chemical properties and evaluation
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Suppress warnings in RDKit
import warnings
warnings.filterwarnings('ignore')

# Set Seaborn style
sns.set_style('whitegrid')

# Function to load the VAE model
def load_vae_model(device):
    # Load the vocabulary
    with open('vae_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    
    # Initialize the model with the same parameters
    hidden_dim = 256  # Ensure this matches your trained model
    latent_dim = 64   # Ensure this matches your trained model
    
    # Define the VAE class (same as in your training script)
    class VAE(nn.Module):
        def __init__(self, vocab_size: int, hidden_dim: int, latent_dim: int):
            super(VAE, self).__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            self.encoder = nn.GRU(vocab_size, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.GRU(vocab_size + latent_dim, hidden_dim, batch_first=True)
            self.fc_output = nn.Linear(hidden_dim, vocab_size)
        
        def encode(self, x: torch.Tensor) -> tuple:
            _, h = self.encoder(x)
            h = h.squeeze(0)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor, max_length: int) -> torch.Tensor:
            batch_size = z.size(0)
            h = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
            x = torch.zeros(batch_size, 1, self.vocab_size).to(z.device)
            outputs = []

            for _ in range(max_length):
                z_input = z.unsqueeze(1).repeat(1, 1, 1)
                decoder_input = torch.cat([x, z_input], dim=2)
                output, h = self.decoder(decoder_input, h)
                output = self.fc_output(output)
                outputs.append(output)
                x = torch.softmax(output, dim=-1)

            return torch.cat(outputs, dim=1)

        def forward(self, x: torch.Tensor) -> tuple:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, x.size(1)), mu, logvar

    model = VAE(vocab_size, hidden_dim, latent_dim)
    model.load_state_dict(torch.load('vae_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, vocab

# Function to generate molecules using VAE
def generate_smiles_vae(model, vocab, num_samples=10, max_length=100):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    generated_smiles = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, model.latent_dim).to(device)
            x = torch.zeros(1, 1, model.vocab_size).to(device)
            x[0, 0, vocab['<']] = 1
            h = torch.zeros(1, 1, model.hidden_dim).to(device)

            smiles = ''
            for _ in range(max_length):
                z_input = z.unsqueeze(1)
                decoder_input = torch.cat([x, z_input], dim=2)
                output, h = model.decoder(decoder_input, h)
                output = model.fc_output(output)

                probs = torch.softmax(output.squeeze(0), dim=-1)
                next_char = torch.multinomial(probs, 1).item()

                if next_char == vocab['>']:
                    break

                smiles += inv_vocab.get(next_char, '')
                x = torch.zeros(1, 1, model.vocab_size).to(device)
                x[0, 0, next_char] = 1

            generated_smiles.append(smiles)

    return generated_smiles

# Function to post-process and validate SMILES strings
def enhanced_post_process_smiles(smiles: str) -> str:
    smiles = smiles.replace('<', '').replace('>', '')
    allowed_chars = set('CNOPSFIBrClcnops()[]=@+-#0123456789')
    smiles = ''.join(c for c in smiles if c in allowed_chars)

    # Balance parentheses
    open_count = smiles.count('(')
    close_count = smiles.count(')')
    if open_count > close_count:
        smiles += ')' * (open_count - close_count)
    elif close_count > open_count:
        smiles = '(' * (close_count - open_count) + smiles

    # Replace invalid double bonds
    smiles = smiles.replace('==', '=')

    # Attempt to close unclosed rings
    for i in range(1, 10):
        if smiles.count(str(i)) % 2 != 0:
            smiles += str(i)

    return smiles

def validate_and_correct_smiles(smiles: str) -> Tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            return True, Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            pass
    return False, smiles

# Function to analyze molecules
def analyze_molecules(smiles_list: List[str], training_smiles_set: set) -> Dict:
    results = {
        'total': len(smiles_list),
        'valid': 0,
        'invalid': 0,
        'unique': 0,
        'corrected': 0,
        'novel': 0,
        'properties': [],
        'invalid_smiles': []
    }

    unique_smiles = set()
    novel_smiles = set()

    for smiles in smiles_list:
        processed_smiles = enhanced_post_process_smiles(smiles)
        is_valid, corrected_smiles = validate_and_correct_smiles(processed_smiles)

        if is_valid:
            results['valid'] += 1
            unique_smiles.add(corrected_smiles)
            if corrected_smiles != processed_smiles:
                results['corrected'] += 1

            if corrected_smiles not in training_smiles_set:
                novel_smiles.add(corrected_smiles)
                results['novel'] += 1

            mol = Chem.MolFromSmiles(corrected_smiles)
            results['properties'].append({
                'smiles': corrected_smiles,
                'MolWt': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol)
            })
        else:
            results['invalid'] += 1
            results['invalid_smiles'].append(smiles)

    results['unique'] = len(unique_smiles)
    return results

# # Function to visualize molecules
# def visualize_molecules(smiles_list: List[str], n: int = 5) -> Optional[Image.Image]:
#     valid_mols = []
#     invalid_count = 0
#     for i, smiles in enumerate(smiles_list):
#         smiles = smiles.strip().strip('<>').strip()
#         if not smiles:
#             invalid_count += 1
#             continue
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is not None:
#                 valid_mols.append(mol)
#                 if len(valid_mols) == n:
#                     break
#             else:
#                 invalid_count += 1
#         except Exception:
#             invalid_count += 1

#     if not valid_mols:
#         return None

#     try:
#         img = Draw.MolsToGridImage(
#             valid_mols,
#             molsPerRow=min(3, len(valid_mols)),
#             subImgSize=(200, 200),
#             legends=[f"Mol {i+1}" for i in range(len(valid_mols))]
#         )
#         return img
#     except Exception:
#         return None


# def visualize_molecules(smiles_list: List[str], n: int = 5) -> Optional[str]:
#     valid_mols = []
#     invalid_count = 0
#     for i, smiles in enumerate(smiles_list):
#         smiles = smiles.strip().strip('<>').strip()
#         if not smiles:
#             invalid_count += 1
#             continue
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is not None:
#                 valid_mols.append(mol)
#                 if len(valid_mols) == n:
#                     break
#             else:
#                 invalid_count += 1
#         except Exception:
#             invalid_count += 1

#     if not valid_mols:
#         return None

#     try:
#         # Use SVG rendering instead of PIL
#         legends = [f"Mol {i+1}" for i in range(len(valid_mols))]
#         drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)  # Single large SVG
#         drawer.DrawMolecules(valid_mols, legends=legends)
#         drawer.FinishDrawing()
#         svg = drawer.GetDrawingText()
#         return svg
#     except Exception as e:
#         print(f"Error in visualization: {e}")
#         return None


# Streamlit app
def main():
    st.set_page_config(
        page_title="Molecule Generator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧪 Molecular Generation and Analysis using VAE")
    st.markdown("""
    This application allows you to generate novel molecular structures using a Variational Autoencoder (VAE) model trained on the QM9 dataset.
    You can generate molecules, visualize them, and explore their chemical properties.
    """)

    # Sidebar configuration
    st.sidebar.title("Configuration")
    st.sidebar.markdown("Adjust the settings below to generate molecules.")

    # Load training data
    df = pd.read_csv("https://raw.githubusercontent.com/urchade/molgen/master/qm9.csv")
    smiles_list = df['smiles'].tolist()
    training_smiles_set = set(smiles_list)

    # Number of samples
    num_samples = st.sidebar.slider("Number of Molecules to Generate", min_value=5, max_value=500, value=50, step=5)

    # Random seed
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate molecules button
    if st.sidebar.button("Generate Molecules"):
        st.info("Generating molecules...")

        # Load VAE model
        model, vocab = load_vae_model(device)
        progress_bar = st.progress(0)
        generated_smiles = []

        # Batch generation for progress tracking
        batch_size = 10
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            smiles_batch = generate_smiles_vae(model, vocab, num_samples=batch_size)
            generated_smiles.extend(smiles_batch)
            progress_bar.progress((i + 1) / num_batches)

        # Analyze molecules
        analysis = analyze_molecules(generated_smiles, training_smiles_set)

        # Display summary
        st.success("Molecule generation completed!")
        st.subheader("Summary of Generated Molecules")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Generated", analysis['total'])
        col2.metric("Valid Molecules", f"{analysis['valid']} ({analysis['valid']/analysis['total']:.2%})")
        col3.metric("Unique Molecules", f"{analysis['unique']} ({analysis['unique']/analysis['total']:.2%})")

        col1, col2 = st.columns(2)
        col1.metric("Novel Molecules", f"{analysis['novel']} ({analysis['novel']/analysis['total']:.2%})")
        col2.metric("Corrected Molecules", f"{analysis['corrected']} ({analysis['corrected']/analysis['total']:.2%})")

        # Display properties
        if analysis['properties']:
            st.subheader("Properties of Generated Molecules")
            df_properties = pd.DataFrame(analysis['properties'])
            st.dataframe(df_properties.style.highlight_max(axis=0))

            # Property distributions
            st.subheader("Property Distributions")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            sns.histplot(df_properties['MolWt'], bins=20, ax=axs[0, 0], kde=True, color='skyblue')
            axs[0, 0].set_title('Molecular Weight Distribution')

            sns.histplot(df_properties['LogP'], bins=20, ax=axs[0, 1], kde=True, color='salmon')
            axs[0, 1].set_title('LogP Distribution')

            sns.histplot(df_properties['NumHDonors'], bins=range(0, df_properties['NumHDonors'].max() + 2), ax=axs[1, 0], kde=False, color='limegreen')
            axs[1, 0].set_title('Number of H Donors')

            sns.histplot(df_properties['NumHAcceptors'], bins=range(0, df_properties['NumHAcceptors'].max() + 2), ax=axs[1, 1], kde=False, color='violet')
            axs[1, 1].set_title('Number of H Acceptors')

            plt.tight_layout()
            st.pyplot(fig)

            # # Visualize molecules
            # st.subheader("Sample Molecules")
            # mol_image = visualize_molecules([prop['smiles'] for prop in analysis['properties']], n=9)
            # if mol_image:
            #     st.image(mol_image)
            # else:
            #     st.write("No valid molecules to display.")

            # st.subheader("Sample Molecules")
            # svg = visualize_molecules([prop['smiles'] for prop in analysis['properties']], n=9)
            # if svg:
            #     st.components.v1.html(svg, height=800)
            # else:
            #     st.write("No valid molecules to display.")

            # Download option
            csv = df_properties.to_csv(index=False)
            st.download_button(
                label="💾 Download SMILES and Properties as CSV",
                data=csv,
                file_name='generated_molecules.csv',
                mime='text/csv'
            )
        else:
            st.warning("No valid molecules were generated.")
    else:
        st.info("Click 'Generate Molecules' in the sidebar to start.")

    # About section
    st.sidebar.title("About")
    st.sidebar.info("""
    **Molecule Generator App**

    This app uses a Variational Autoencoder (VAE) model trained on the QM9 dataset to generate novel molecular structures.

    - **Developed by**: ARJUN,KAUSTUBH,NACHIKET
    - **GitHub**: https://github.com/arjuntyagi19/MLOps_final_year_project
    """)

    # Hide Streamlit style elements
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
