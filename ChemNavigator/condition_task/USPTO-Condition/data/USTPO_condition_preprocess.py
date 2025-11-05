
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Update these paths to match your file locations
INPUT_CSV_PATH = 'USPTO_condition.csv'
OUTPUT_CSV_PATH = 'USPTO_condition_fingerprint.csv'
CHEM_DICT_FILE = 'chemical_to_id.json'
ID_CHEM_DICT_FILE = 'id_to_chemical.json'

# Fingerprint structure constants
FINGERPRINT_LENGTH = 640
VALUES_PER_STEP = 32
DEFAULT_TEMP = -1.0
DEFAULT_DURATION = -1.0


# --- Dictionary Management ---
def load_or_initialize_dictionaries():
    """Loads dictionaries or initializes them if they don't exist."""
    if os.path.exists(CHEM_DICT_FILE):
        with open(CHEM_DICT_FILE, 'r') as f:
            chemical_to_id = json.load(f)
        with open(ID_CHEM_DICT_FILE, 'r') as f:
            id_to_chemical = {int(k): v for k, v in json.load(f).items()}
    else:
        # Initialize with a padding token
        chemical_to_id = {'[PAD]': 0}
        id_to_chemical = {0: '[PAD]'}
    return chemical_to_id, id_to_chemical


def save_dictionaries(chemical_to_id, id_to_chemical):
    """Saves dictionaries to JSON files."""
    os.makedirs(os.path.dirname(CHEM_DICT_FILE), exist_ok=True)
    with open(CHEM_DICT_FILE, 'w') as f:
        json.dump(chemical_to_id, f, indent=4)
    with open(ID_CHEM_DICT_FILE, 'w') as f:
        json.dump(id_to_chemical, f, indent=4)


def get_or_assign_id(item_name, item_to_id_dict, id_to_item_dict):
    """Gets existing ID or assigns a new one for a chemical."""
    if pd.isna(item_name) or not str(item_name).strip():
        return 0  # Return padding ID for empty items

    # Standardize name for dictionary lookup
    item_name = str(item_name).strip()

    if item_name not in item_to_id_dict:
        new_id = len(item_to_id_dict)
        item_to_id_dict[item_name] = new_id
        id_to_item_dict[new_id] = item_name
        return new_id
    return item_to_id_dict[item_name]


# --- Main Processing Function ---
def create_fingerprints_for_new_dataset(input_path, output_path):
    """
    Reads the new dataset format, creates condition fingerprints, and saves a processed CSV.
    """
    print(f"Loading new dataset from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_path}")
        return

    # Check for required columns
    required_cols = ['canonical_rxn', 'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2', 'dataset']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain the following columns: {required_cols}")
        return

    chemical_to_id, id_to_chemical = load_or_initialize_dictionaries()

    processed_data = []

    print("Processing rows and generating fingerprints...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # --- Initialize Fingerprint ---
        # All steps are inactive by default
        fingerprint = np.zeros(FINGERPRINT_LENGTH, dtype=np.float32)

        # --- Activate and Populate Step 1 ---
        # The new data format corresponds to a single reaction step.
        # We will populate the first 32-value segment of the fingerprint.
        step_offset = 0

        # 1. Activate Step
        fingerprint[step_offset + 0] = 1.0  # is_active

        # 2. Get chemical IDs, adding new ones to the dictionary if needed.
        r1_id = get_or_assign_id(row['reagent1'], chemical_to_id, id_to_chemical)
        r2_id = get_or_assign_id(row['reagent2'], chemical_to_id, id_to_chemical)
        c1_id = get_or_assign_id(row['catalyst1'], chemical_to_id, id_to_chemical)
        s1_id = get_or_assign_id(row['solvent1'], chemical_to_id, id_to_chemical)
        s2_id = get_or_assign_id(row['solvent2'], chemical_to_id, id_to_chemical)

        # 3. Populate fingerprint vector at the correct positions
        # Reagents (5 slots, indices 1-5)
        fingerprint[step_offset + 1] = r1_id
        fingerprint[step_offset + 2] = r2_id
        # Other reagent slots (3, 4, 5) remain 0

        # Catalysts (4 slots, indices 6-9)
        fingerprint[step_offset + 6] = c1_id
        # Other catalyst slots (7, 8, 9) remain 0

        # Solvents (4 slots, indices 10-13)
        fingerprint[step_offset + 10] = s1_id
        fingerprint[step_offset + 11] = s2_id
        # Other solvent slots (12, 13) remain 0

        # Temp-Time Segments (16 slots, indices 14-29)
        # Fill with default values as this info is not in the new dataset
        temp_time_defaults = [DEFAULT_TEMP, DEFAULT_TEMP, DEFAULT_DURATION, 0.0] * 4
        fingerprint[step_offset + 14: step_offset + 30] = temp_time_defaults

        # Other Conditions (2 slots, indices 30-31) remain 0

        # --- Process SMILES ---
        try:
            reactants_smiles, products_smiles = row['canonical_rxn'].split('>>')
        except ValueError:
            print(f"Warning: Skipping row due to invalid canonical_rxn format: {row['canonical_rxn']}")
            continue

        processed_data.append({
            'reactants': reactants_smiles,
            'products': products_smiles,
            'condition_fingerprint': json.dumps(fingerprint.tolist()),
            'dataset': row['dataset']
        })

    # --- Save Processed Data and Dictionaries ---
    output_df = pd.DataFrame(processed_data)
    output_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

    save_dictionaries(chemical_to_id, id_to_chemical)
    print(f"Chemical dictionaries saved.")
    print(f"Total chemicals in dictionary: {len(chemical_to_id)}")


if __name__ == '__main__':

    create_fingerprints_for_new_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH)