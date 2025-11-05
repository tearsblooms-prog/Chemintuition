import pandas as pd
import re
import json
import os
from collections import defaultdict
# These constants define the structure of the fingerprint and can be adjusted if needed.
MAX_STAGES = 2
MAX_STEPS_PER_STAGE = 10
TOTAL_POTENTIAL_STEPS = MAX_STAGES * MAX_STEPS_PER_STAGE

NUM_REAGENTS = 5
NUM_CATALYSTS = 4
NUM_SOLVENTS = 4
NUM_TEMP_TIME_SEGMENTS = 4
NUM_VALUES_PER_SEGMENT = 4  # T_start, T_end, duration_min, is_reflux
NUM_OTHER_CONDITIONS = 2

# Values per step: 1 (active) + R(5) + C(4) + S(4) + TT_Segments(4*4=16) + Other(2) = 32
VALUES_PER_STEP = 1 + NUM_REAGENTS + NUM_CATALYSTS + NUM_SOLVENTS + \
                  (NUM_TEMP_TIME_SEGMENTS * NUM_VALUES_PER_SEGMENT) + \
                  NUM_OTHER_CONDITIONS
FINGERPRINT_LENGTH = TOTAL_POTENTIAL_STEPS * VALUES_PER_STEP

DEFAULT_TEMP = -1.0  # For unspecified temperatures
DEFAULT_DURATION = -1  # For unspecified durations or transitions
RT_TEMP = 25.0

# Dictionaries file paths
CHEM_DICT_FILE = './chemical_to_id.json'
ID_CHEM_DICT_FILE = './id_to_chemical.json'
OTHER_DICT_FILE = './other_condition_to_id.json'
ID_OTHER_DICT_FILE = './id_to_other_condition.json'


# --- Dictionary Management ---
def load_dictionaries():
    """Loads dictionaries from JSON files."""
    chemical_to_id = {}
    id_to_chemical = {}
    other_condition_to_id = {}
    id_to_other_condition = {}

    if os.path.exists(CHEM_DICT_FILE):
        with open(CHEM_DICT_FILE, 'r') as f:
            chemical_to_id = json.load(f)
    if os.path.exists(ID_CHEM_DICT_FILE):
        with open(ID_CHEM_DICT_FILE, 'r') as f:
            id_to_chemical = {int(k): v for k, v in json.load(f).items()}

    if os.path.exists(OTHER_DICT_FILE):
        with open(OTHER_DICT_FILE, 'r') as f:
            other_condition_to_id = json.load(f)
    if os.path.exists(ID_OTHER_DICT_FILE):
        with open(ID_OTHER_DICT_FILE, 'r') as f:
            id_to_other_condition = {int(k): v for k, v in json.load(f).items()}

    return chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition


def save_dictionaries(chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition):
    """Saves dictionaries to JSON files."""
    with open(CHEM_DICT_FILE, 'w') as f:
        json.dump(chemical_to_id, f, indent=4)
    with open(ID_CHEM_DICT_FILE, 'w') as f:
        json.dump(id_to_chemical, f, indent=4)
    with open(OTHER_DICT_FILE, 'w') as f:
        json.dump(other_condition_to_id, f, indent=4)
    with open(ID_OTHER_DICT_FILE, 'w') as f:
        json.dump(id_to_other_condition, f, indent=4)


def get_or_assign_id(item_name, item_to_id_dict, id_to_item_dict):
    """Gets existing ID or assigns a new one for an item."""
    # Ensure item_name is a string and stripped of whitespace
    if pd.isna(item_name) or item_name is None:
        return 0
    item_name = str(item_name).strip()

    if not item_name:
        return 0  # 0 is reserved for padding/None
    if item_name not in item_to_id_dict:
        new_id = len(item_to_id_dict) + 1  # Start IDs from 1
        item_to_id_dict[item_name] = new_id
        id_to_item_dict[new_id] = item_name
        return new_id
    return item_to_id_dict[item_name]


# --- NEW: Fingerprint Generation for Structured CSV ---
def row_to_fingerprint(row, chemical_to_id, id_to_chemical):
    """
    Converts a structured DataFrame row directly to a numerical fingerprint.
    Each row is treated as a single step reaction (Step1.1).
    """
    fingerprint = [0.0] * FINGERPRINT_LENGTH

    # --- This reaction happens at Step 1.1 ---
    global_step_idx = 0  # Step 1.1 maps to index 0
    step_offset = global_step_idx * VALUES_PER_STEP

    # 1. Activate the step
    fingerprint[step_offset] = 1.0
    current_offset = step_offset + 1

    # 2. Reagents
    # aryl_halide, amine are reactants. ligand, base, additive are also treated as reagents.
    reagent_names = [row.get('ligand'),
        row.get('base'), row.get('additive')
    ]
    reagent_ids = [get_or_assign_id(name, chemical_to_id, id_to_chemical) for name in reagent_names]
    reagent_ids = [rid for rid in reagent_ids if rid != 0]  # Remove empty/padding IDs
    reagent_ids.extend([0] * (NUM_REAGENTS - len(reagent_ids)))  # Pad to fixed length
    fingerprint[current_offset: current_offset + NUM_REAGENTS] = reagent_ids[:NUM_REAGENTS]
    current_offset += NUM_REAGENTS

    # 3. Catalysts
    catalyst_names = [row.get('catalyst')]
    catalyst_ids = [get_or_assign_id(name, chemical_to_id, id_to_chemical) for name in catalyst_names]
    catalyst_ids = [cid for cid in catalyst_ids if cid != 0]
    catalyst_ids.extend([0] * (NUM_CATALYSTS - len(catalyst_ids)))
    fingerprint[current_offset: current_offset + NUM_CATALYSTS] = catalyst_ids[:NUM_CATALYSTS]
    current_offset += NUM_CATALYSTS

    # 4. Solvents
    solvent_ids = [0] * NUM_SOLVENTS
    fingerprint[current_offset: current_offset + NUM_SOLVENTS] = solvent_ids
    current_offset += NUM_SOLVENTS

    # 5. Temp-Time Segments
    for _ in range(NUM_TEMP_TIME_SEGMENTS):
        fingerprint[current_offset] = DEFAULT_TEMP  # T_start
        fingerprint[current_offset + 1] = DEFAULT_TEMP  # T_end
        fingerprint[current_offset + 2] = DEFAULT_DURATION  # duration_min
        fingerprint[current_offset + 3] = 0.0  # is_reflux
        current_offset += NUM_VALUES_PER_SEGMENT

    # 6. Other Conditions
    other_ids = [0] * NUM_OTHER_CONDITIONS
    fingerprint[current_offset: current_offset + NUM_OTHER_CONDITIONS] = other_ids
    current_offset += NUM_OTHER_CONDITIONS

    return fingerprint


# --- Fingerprint Decoding  ---
def format_temperature(temp_val):
    if temp_val == DEFAULT_TEMP: return ""
    if temp_val == RT_TEMP: return "rt"
    return f"{temp_val}°C"


def format_duration(duration_val_min):
    if duration_val_min == DEFAULT_DURATION or duration_val_min == 0: return ""
    if duration_val_min == 8.11 * 60.0: return "overnight"
    if duration_val_min >= 60 and duration_val_min % 60 == 0:
        return f"{int(duration_val_min / 60)} h"
    return f"{int(duration_val_min)} min"


def fingerprint_to_structured_condition(fingerprint, id_to_chemical, id_to_other_condition):
    """Converts a numerical fingerprint back to a human-readable string."""
    if not isinstance(fingerprint, list):
        try:
            fingerprint = json.loads(fingerprint)
        except:
            fingerprint = [float(x) for x in fingerprint.strip('[]').split(',')]

    decoded_steps = []
    for global_step_idx in range(TOTAL_POTENTIAL_STEPS):
        step_offset = global_step_idx * VALUES_PER_STEP

        if fingerprint[step_offset] == 1.0:
            stage_num = (global_step_idx // MAX_STEPS_PER_STAGE) + 1
            step_in_stage_num = (global_step_idx % MAX_STEPS_PER_STAGE) + 1
            step_label = f"Step{stage_num}.{step_in_stage_num}:"

            parts = [step_label]
            current_offset = step_offset + 1

            # Reagents
            r_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_REAGENTS]]
            current_offset += NUM_REAGENTS
            if any(r_id > 0 for r_id in r_ids):
                parts.append("Reagents:" + "#".join([id_to_chemical.get(r_id, "") for r_id in r_ids if r_id > 0]))

            # Catalysts
            c_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_CATALYSTS]]
            current_offset += NUM_CATALYSTS
            if any(c_id > 0 for c_id in c_ids):
                parts.append("Catalysts:" + "#".join([id_to_chemical.get(c_id, "") for c_id in c_ids if c_id > 0]))

            # Solvents
            s_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_SOLVENTS]]
            current_offset += NUM_SOLVENTS
            if any(s_id > 0 for s_id in s_ids):
                parts.append("Solvents:" + "#".join([id_to_chemical.get(s_id, "") for s_id in s_ids if s_id > 0]))

            # Temp-Time Segments
            segment_strings = []
            for i in range(NUM_TEMP_TIME_SEGMENTS):
                t_start, t_end, duration, is_reflux = fingerprint[current_offset:current_offset + 4]
                current_offset += NUM_VALUES_PER_SEGMENT
                if t_start != DEFAULT_TEMP or t_end != DEFAULT_TEMP or duration != DEFAULT_DURATION or is_reflux == 1.0:
                    seg_parts = []
                    if t_start != DEFAULT_TEMP and t_start == t_end:
                        seg_parts.append(format_temperature(t_start))
                    elif t_start != DEFAULT_TEMP and t_start != t_end:
                        seg_parts.append(f"{format_temperature(t_start)} → {format_temperature(t_end)}")

                    formatted_dur = format_duration(duration)
                    if formatted_dur: seg_parts.append(formatted_dur)
                    if is_reflux == 1.0: seg_parts.append("reflux")
                    if seg_parts: segment_strings.append(", ".join(seg_parts))

            if segment_strings: parts.append("; ".join(segment_strings))

            # Other Conditions
            o_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_OTHER_CONDITIONS]]
            if any(o_id > 0 for o_id in o_ids):
                parts.append("; ".join([id_to_other_condition.get(o_id, "") for o_id in o_ids if o_id > 0]))

            # Join parts for the final step description string
            step_desc = parts[0] + " "
            component_strings = [p.strip() for p in parts[1:] if p.strip()]
            step_desc += "; ".join(component_strings)
            decoded_steps.append(step_desc)

    return "\n".join(decoded_steps)


# --- Main Processing Function for the new CSV format ---
def process_new_format_csv(input_csv_path, output_csv_path):
    """
    Reads the new CSV format, generates fingerprints directly from columns,
    and saves to a new CSV.
    """
    chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition = load_dictionaries()

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Verify required columns exist
    required_columns = ['aryl_halide', 'amine', 'catalyst', 'ligand', 'base', 'additive']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: One or more required columns not found in the CSV.")
        print(f"Required columns: {required_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    fingerprints = []
    decoded_conditions = []

    for index, row in df.iterrows():
        try:
            fp = row_to_fingerprint(row, chemical_to_id, id_to_chemical)
            decoded = fingerprint_to_structured_condition(fp, id_to_chemical, id_to_other_condition)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            fp = [0.0] * FINGERPRINT_LENGTH  # Error fingerprint
            decoded = f"ERROR_PARSING: {e}"

        fingerprints.append(json.dumps(fp))  # Store fingerprint as a JSON string
        decoded_conditions.append(decoded)

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} rows...")

    # Add the new columns to the DataFrame
    df['reaction_fingerprint'] = fingerprints
    df['decoded_condition'] = decoded_conditions

    try:
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully processed and saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    save_dictionaries(chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition)
    print("Dictionaries updated and saved.")


# --- Example Usage ---
if __name__ == '__main__':
    # Path to your new CSV file
    input_csv = 'buchwald_hartwig_processed.csv'

    # Path for the output file with the generated fingerprints
    output_csv = 'buchwald_hartwig_processed_with_fingerprints.csv'

    # Process the new CSV format
    process_new_format_csv(input_csv, output_csv)