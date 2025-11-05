import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm
from pathlib import Path


def canonicalize_with_dict(smi: str, can_smi_dict: dict) -> str:
    """
    Canonicalize SMILES strings and cache them in a dictionary for efficiency.

    Args:
        smi (str): Input SMILES string.
        can_smi_dict (dict): Dictionary used to cache canonicalized SMILES.

    Returns:
        str: Canonicalized SMILES string.
    """
    if smi not in can_smi_dict:
        # If the SMILES is not in the cache, canonicalize it and store it
        can_smi_dict[smi] = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    return can_smi_dict[smi]


def process_buchwald_hartwig_to_csv(input_excel_path: str, output_csv_path: str):
    """
    Process Buchwald–Hartwig reaction data and save it as a CSV file.

    Args:
        input_excel_path (str): Path to the input Excel file.
        output_csv_path (str): Path to the output CSV file.
    """
    # Ensure the output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    print(f"Reading data from {input_excel_path} ...")
    df = pd.read_excel(input_excel_path, sheet_name='FullCV_01')

    # Optionally perform Z-score normalization for yield (Output)
    # df['Output'] = (df['Output'] - df.Output.mean()) / df.Output.std()

    # 2. Define reaction template and fixed reactants
    # SMARTS template for the Buchwald–Hartwig coupling reaction
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    # Fixed reactant: p-toluidine
    methylaniline = 'Cc1ccc(N)cc1'
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    # Fixed palladium catalyst
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))

    # Create the reaction object based on the template
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)

    # 3. Compute reaction products
    print("Computing reaction products ...")
    products = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Product computation"):
        # Reactants: “Aryl halide” in each row and the fixed p-toluidine
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)
        # Extract product SMILES
        rxn_products_smiles = {Chem.MolToSmiles(mol[0]) for mol in rxn_products}
        # Ensure each reaction generates exactly one product
        assert len(rxn_products_smiles) == 1, f"Reaction generated multiple or zero products: {rxn_products_smiles}"
        products.append(list(rxn_products_smiles)[0])

    # Add product SMILES to DataFrame
    df['product'] = products

    # 4. Canonicalize and organize all components
    print("Canonicalizing and organizing SMILES ...")
    rxn_smiles_lists = []
    can_smiles_dict = {}  # Cache for canonicalized SMILES

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="SMILES canonicalization"):
        # Canonicalize each variable component
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)

        # Build the list containing all components and yield
        rxn_smiles_list = [
            aryl_halide,   # Aryl halide (Reactant 1)
            methylaniline, # p-Toluidine (Reactant 2)
            pd_catalyst,   # Pd catalyst
            ligand,        # Ligand
            base,          # Base
            additive,      # Additive
            row['product'],# Product
            row['Output']  # Yield
        ]
        rxn_smiles_lists.append(rxn_smiles_list)

    # 5. Save to CSV file
    # Define CSV column names
    column_names = [
        'aryl_halide',
        'amine',
        'catalyst',
        'ligand',
        'base',
        'additive',
        'product',
        'output'
    ]

    # Create a new DataFrame and save
    output_df = pd.DataFrame(rxn_smiles_lists, columns=column_names)
    output_df.to_csv(output_csv_path, index=False)

    print(f"Processing completed! Data successfully saved to {output_csv_path}")


# --- Main Entry Point ---
if __name__ == '__main__':
    # Define input and output file paths
    # Assume the Excel file is located in the 'data' folder in the same directory as the script
    INPUT_FILE = 'Dreher_and_Doyle_input_data.xlsx'

    # Define output CSV file path
    OUTPUT_FILE = 'buchwald_hartwig/buchwald_hartwig_processed.csv'
    process_buchwald_hartwig_to_csv(input_excel_path=str(INPUT_FILE), output_csv_path=str(OUTPUT_FILE))
