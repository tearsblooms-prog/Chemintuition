import pandas as pd
import json
import os

MAX_STAGES = 2
MAX_STEPS_PER_STAGE = 10
TOTAL_POTENTIAL_STEPS = MAX_STAGES * MAX_STEPS_PER_STAGE

NUM_REAGENTS = 5
NUM_CATALYSTS = 4
NUM_SOLVENTS = 4
NUM_TEMP_TIME_SEGMENTS = 4
NUM_VALUES_PER_SEGMENT = 4  # T_start, T_end, duration_min, is_reflux
NUM_OTHER_CONDITIONS = 2

VALUES_PER_STEP = 1 + NUM_REAGENTS + NUM_CATALYSTS + NUM_SOLVENTS + \
                  (NUM_TEMP_TIME_SEGMENTS * NUM_VALUES_PER_SEGMENT) + \
                  NUM_OTHER_CONDITIONS
FINGERPRINT_LENGTH = TOTAL_POTENTIAL_STEPS * VALUES_PER_STEP

DEFAULT_TEMP = -1.0
DEFAULT_DURATION = -1.0
RT_TEMP = 25.0

DATA_DIR = './data'
CHEM_DICT_FILE = os.path.join(DATA_DIR, 'suzuki_chemical_to_id.json')
ID_CHEM_DICT_FILE = os.path.join(DATA_DIR, 'suzuki_id_to_chemical.json')
OTHER_DICT_FILE = os.path.join(DATA_DIR, 'suzuki_other_condition_to_id.json')
ID_OTHER_DICT_FILE = os.path.join(DATA_DIR, 'suzuki_id_to_other_condition.json')


# --- Dictionary Management ---
def load_dictionaries():
    """Load dictionaries from JSON files. Create empty ones if not found."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    chemical_to_id = {}
    id_to_chemical = {}
    other_condition_to_id = {}
    id_to_other_condition = {}

    if os.path.exists(CHEM_DICT_FILE):
        with open(CHEM_DICT_FILE, 'r', encoding='utf-8') as f:
            chemical_to_id = json.load(f)
    if os.path.exists(ID_CHEM_DICT_FILE):
        with open(ID_CHEM_DICT_FILE, 'r', encoding='utf-8') as f:
            id_to_chemical = {int(k): v for k, v in json.load(f).items()}

    if os.path.exists(OTHER_DICT_FILE):
        with open(OTHER_DICT_FILE, 'r', encoding='utf-8') as f:
            other_condition_to_id = json.load(f)
    if os.path.exists(ID_OTHER_DICT_FILE):
        with open(ID_OTHER_DICT_FILE, 'r', encoding='utf-8') as f:
            id_to_other_condition = {int(k): v for k, v in json.load(f).items()}

    return chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition


def save_dictionaries(chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition):
    """Save dictionaries to JSON files."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open(CHEM_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chemical_to_id, f, indent=4, ensure_ascii=False)
    with open(ID_CHEM_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_to_chemical, f, indent=4, ensure_ascii=False)
    with open(OTHER_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(other_condition_to_id, f, indent=4, ensure_ascii=False)
    with open(ID_OTHER_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_to_other_condition, f, indent=4, ensure_ascii=False)


def get_or_assign_id(item_name, item_to_id_dict, id_to_item_dict):
    """Get existing ID or assign a new one."""
    if not isinstance(item_name, str):
        return 0
    item_name = item_name.strip()
    if not item_name:
        return 0
    if item_name not in item_to_id_dict:
        new_id = len(item_to_id_dict) + 1
        item_to_id_dict[item_name] = new_id
        id_to_item_dict[new_id] = item_name
        return new_id
    return item_to_id_dict[item_name]


# --- Fingerprint Generation ---
def generate_fingerprint_from_row(row, chemical_to_id, id_to_chemical):
    """Generate a numerical fingerprint from a row of structured data."""
    fingerprint = [0.0] * FINGERPRINT_LENGTH

    global_step_idx = 0
    step_offset = global_step_idx * VALUES_PER_STEP

    fingerprint[step_offset] = 1.0
    current_offset = step_offset + 1

    reagents = [row.get('reagent')]
    r_ids = [get_or_assign_id(r, chemical_to_id, id_to_chemical) for r in reagents if r and pd.notna(r)]
    r_ids.extend([0] * (NUM_REAGENTS - len(r_ids)))
    fingerprint[current_offset: current_offset + NUM_REAGENTS] = r_ids[:NUM_REAGENTS]
    current_offset += NUM_REAGENTS

    catalysts = [row.get('catalyst')]
    c_ids = [get_or_assign_id(c, chemical_to_id, id_to_chemical) for c in catalysts if c and pd.notna(c)]
    c_ids.extend([0] * (NUM_CATALYSTS - len(c_ids)))
    fingerprint[current_offset: current_offset + NUM_CATALYSTS] = c_ids[:NUM_CATALYSTS]
    current_offset += NUM_CATALYSTS

    solvents = [row.get('solvent')]
    s_ids = [get_or_assign_id(s, chemical_to_id, id_to_chemical) for s in solvents if s and pd.notna(s)]
    s_ids.extend([0] * (NUM_SOLVENTS - len(s_ids)))
    fingerprint[current_offset: current_offset + NUM_SOLVENTS] = s_ids[:NUM_SOLVENTS]
    current_offset += NUM_SOLVENTS

    for _ in range(NUM_TEMP_TIME_SEGMENTS):
        fingerprint[current_offset] = DEFAULT_TEMP
        fingerprint[current_offset + 1] = DEFAULT_TEMP
        fingerprint[current_offset + 2] = DEFAULT_DURATION
        fingerprint[current_offset + 3] = 0.0
        current_offset += NUM_VALUES_PER_SEGMENT

    return fingerprint


# --- Fingerprint Decoding ---
def fingerprint_to_structured_condition(fingerprint, id_to_chemical, id_to_other_condition):
    """Decode numerical fingerprint back to structured condition string."""
    if not isinstance(fingerprint, list):
        try:
            fingerprint = json.loads(fingerprint)
        except (json.JSONDecodeError, TypeError):
            fingerprint = [float(x) for x in str(fingerprint).strip('[]').split(',') if x.strip()]

    decoded_steps = []
    for global_step_idx in range(TOTAL_POTENTIAL_STEPS):
        step_offset = global_step_idx * VALUES_PER_STEP

        if fingerprint[step_offset] == 1.0:
            stage_num = (global_step_idx // MAX_STEPS_PER_STAGE) + 1
            step_in_stage_num = (global_step_idx % MAX_STEPS_PER_STAGE) + 1
            step_label = f"Step{stage_num}.{step_in_stage_num}:"

            parts = [step_label]
            current_offset = step_offset + 1

            r_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_REAGENTS]]
            current_offset += NUM_REAGENTS
            if any(r_id > 0 for r_id in r_ids):
                parts.append("Reagents:" + "#".join([id_to_chemical.get(str(r_id), f"ID_{r_id}_NOT_FOUND") for r_id in r_ids if r_id > 0]))

            c_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_CATALYSTS]]
            current_offset += NUM_CATALYSTS
            if any(c_id > 0 for c_id in c_ids):
                parts.append("Catalysts:" + "#".join([id_to_chemical.get(str(c_id), f"ID_{c_id}_NOT_FOUND") for c_id in c_ids if c_id > 0]))

            s_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_SOLVENTS]]
            current_offset += NUM_SOLVENTS
            if any(s_id > 0 for s_id in s_ids):
                parts.append("Solvents:" + "#".join([id_to_chemical.get(str(s_id), f"ID_{s_id}_NOT_FOUND") for s_id in s_ids if s_id > 0]))

            step_desc = parts[0] + " "
            component_strings = [p.strip() for p in parts[1:] if p.strip()]
            step_desc += "; ".join(component_strings)
            decoded_steps.append(step_desc)

    return "\n".join(decoded_steps)


# --- Main Processing ---
def process_suzuki_csv(input_csv_path, output_csv_path):
    """Read input CSV, generate fingerprints and decoded conditions, then save."""
    chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition = load_dictionaries()

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        return
    except Exception:
        return

    fingerprints = []
    decoded_conditions = []

    for index, row in df.iterrows():
        try:
            fp = generate_fingerprint_from_row(row, chemical_to_id, id_to_chemical)
            str_id_to_chemical = {str(k): v for k, v in id_to_chemical.items()}
            str_id_to_other = {str(k): v for k, v in id_to_other_condition.items()}
            decoded = fingerprint_to_structured_condition(fp, str_id_to_chemical, str_id_to_other)
        except Exception as e:
            fp = [0.0] * FINGERPRINT_LENGTH
            decoded = f"ERROR_PARSING: {e}"

        fingerprints.append(json.dumps(fp))
        decoded_conditions.append(decoded)

    df['condition_fingerprint'] = fingerprints
    df['decoded_condition'] = decoded_conditions

    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
    except Exception:
        pass

    save_dictionaries(chemical_to_id, id_to_chemical, other_condition_to_id, id_to_other_condition)


if __name__ == '__main__':
    input_csv = 'suzuki_miyaura_smiles.csv'
    output_csv = 'suzuki_miyaura_smiles_with_fingerprint.csv'
    process_suzuki_csv(input_csv, output_csv)
