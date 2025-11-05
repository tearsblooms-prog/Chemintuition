import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV_PATH = 'data/Reaxys_total_syn_condition.csv'
OUTPUT_CSV_PATH = 'data/Reaxys_total_syn_condition_fingerprint.csv'
CHEM_DICT_FILE = 'data/chemical_to_id.json'
ID_CHEM_DICT_FILE = 'data/id_to_chemical.json'

# Fingerprint structure constants
FINGERPRINT_LENGTH = 640
VALUES_PER_STEP = 32
DEFAULT_TEMP = -1.0
DEFAULT_DURATION = -1.0

def load_or_initialize_dictionaries():
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

    required_cols = ['canonical_rxn', 'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2', 'temperature',
                     'dataset']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Error: Input CSV must contain the following columns: {required_cols}")
        print(f"Missing columns found in the CSV file: {missing_cols}")
        return

    chemical_to_id, id_to_chemical = load_or_initialize_dictionaries()

    processed_data = []

    print("Processing rows and generating fingerprints...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):

        fingerprint = np.zeros(FINGERPRINT_LENGTH, dtype=np.float32)
        step_offset = 0

        # 1. Activate Step
        fingerprint[step_offset + 0] = 1.0  # is_active

        # 2. Get chemical IDs, adding new ones to the dictionary if needed.
        r1_id = get_or_assign_id(row.get('reagent1'), chemical_to_id, id_to_chemical)
        r2_id = get_or_assign_id(row.get('reagent2'), chemical_to_id, id_to_chemical)
        c1_id = get_or_assign_id(row.get('catalyst1'), chemical_to_id, id_to_chemical)
        s1_id = get_or_assign_id(row.get('solvent1'), chemical_to_id, id_to_chemical)
        s2_id = get_or_assign_id(row.get('solvent2'), chemical_to_id, id_to_chemical)

        # 3. Populate fingerprint vector at the correct positions
        # Reagents (5 slots, indices 1-5)
        fingerprint[step_offset + 1] = r1_id
        fingerprint[step_offset + 2] = r2_id

        # Catalysts (4 slots, indices 6-9)
        fingerprint[step_offset + 6] = c1_id

        # Solvents (4 slots, indices 10-13)
        fingerprint[step_offset + 10] = s1_id
        fingerprint[step_offset + 11] = s2_id

        # Temp-Time Segments (16 slots, indices 14-29)
        temp_time_vector = [DEFAULT_TEMP, DEFAULT_TEMP, DEFAULT_DURATION, 0.0] * 4

        try:
            if pd.notna(row['temperature']):
                temp_val = float(row['temperature'])
                temp_time_vector[0] = temp_val  # Seg1_T_start (at index 14)
                temp_time_vector[1] = temp_val  # Seg1_T_end (at index 15)
        except (ValueError, TypeError):
            pass

        fingerprint[step_offset + 14: step_offset + 30] = temp_time_vector

        try:
            reactants_smiles, products_smiles = str(row['canonical_rxn']).split('>>')
        except ValueError:
            print(f"Warning: Skipping row due to invalid canonical_rxn format: {row['canonical_rxn']}")
            continue

        processed_data.append({
            'reactants': reactants_smiles,
            'products': products_smiles,
            'condition_fingerprint': json.dumps(fingerprint.tolist()),
            'dataset': row['dataset']
        })

    output_df = pd.DataFrame(processed_data)
    output_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

    save_dictionaries(chemical_to_id, id_to_chemical)
    print(f"Chemical dictionaries saved.")
    print(f"Total chemicals in dictionary: {len(chemical_to_id)}")


if __name__ == '__main__':

    create_fingerprints_for_new_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH)