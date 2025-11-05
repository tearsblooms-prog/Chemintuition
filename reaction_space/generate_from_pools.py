import pandas as pd
import itertools
from rdkit import Chem
from rdkit.Chem import rdmolops
from functools import lru_cache
import os


# --- Helper Functions (from original script) ---

@lru_cache(maxsize=None)
def get_atom_count(smiles: str) -> int:
    if not isinstance(smiles, str) or not smiles.strip(): return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    return mol.GetNumAtoms()


@lru_cache(maxsize=None)
def get_atom_types(smiles: str) -> set:
    if not isinstance(smiles, str) or not smiles.strip(): return set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


@lru_cache(maxsize=None)
def get_formal_charge(smiles: str) -> int:
    """Calculates the total formal charge of a molecule."""
    if not isinstance(smiles, str) or not smiles.strip(): return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    return rdmolops.GetFormalCharge(mol)


def normalize_reactant_pair(r1: str, r2: str) -> tuple:
    s1 = r1 if isinstance(r1, str) and r1.strip() else ''
    s2 = r2 if isinstance(r2, str) and r2.strip() else ''
    if not s1: return (s2, '')
    if not s2: return (s1, '')
    return tuple(sorted([s1, s2]))


# --- Core Generation Function ---

def generate_candidates_from_pools(original_data_csv: str,
                                   reactants_pool_csv: str,
                                   products_pool_csv: str,
                                   output_csv: str):
    print("--- Step 2: Starting candidate generation from pools ---")

    # 1. Load Fragment Pools
    try:
        df_reactants = pd.read_csv(reactants_pool_csv)
        top_reactants = list(df_reactants['smiles'])

        df_products = pd.read_csv(products_pool_csv)
        top_products = list(df_products['smiles'])
    except FileNotFoundError as e:
        print(f"Error: Pool file not found. Please run '1_build_fragment_pools.py' first.")
        print(e)
        return

    print(f"Loaded {len(top_reactants)} reactants from pool.")
    print(f"Loaded {len(top_products)} products from pool.")

    # 2. Load Original Data for De-duplication
    try:
        df_orig = pd.read_csv(original_data_csv)
    except FileNotFoundError:
        print(f"Error: Original data file not found -> {original_data_csv}")
        return

    existing_reactions = set()
    for _, row in df_orig.iterrows():
        key = (
            normalize_reactant_pair(row.get('reactant1_smiles'), row.get('reactant2_smiles')),
            row.get('product_smiles')
        )
        existing_reactions.add(key)
    print(f"Loaded {len(existing_reactions)} existing reactions for de-duplication.")

    # 3. Create Reactant Combinations (One or two reactants)
    reactant_combos = list(itertools.combinations(top_reactants, 2)) + [(r, '') for r in top_reactants]
    print(f"Generated {len(reactant_combos)} reactant combinations (R1+R2 and R1).")

    # 4. Iterate and Validate
    new_reaction_records = []
    generated_count = 0
    skipped_existing = 0
    skipped_atom_count = 0
    skipped_atom_type = 0
    skipped_charge = 0

    total_combinations = len(reactant_combos) * len(top_products)
    print(f"Starting to check {total_combinations:.2e} theoretical combinations...")

    for i, (r1, r2) in enumerate(reactant_combos):
        norm_pair = normalize_reactant_pair(r1, r2)

        reactant_atoms = get_atom_count(norm_pair[0]) + get_atom_count(norm_pair[1])
        if reactant_atoms == 0: continue

        reactant_types = get_atom_types(norm_pair[0]).union(get_atom_types(norm_pair[1]))
        reactant_charge = get_formal_charge(norm_pair[0]) + get_formal_charge(norm_pair[1])

        for prod_smiles in top_products:
            # Check 1: Already exists
            if (norm_pair, prod_smiles) in existing_reactions:
                skipped_existing += 1
                continue

            product_atoms = get_atom_count(prod_smiles)
            if product_atoms == 0: continue

            # Check 2: Atom count conservation (product atoms <= reactant atoms)
            if product_atoms > reactant_atoms:
                skipped_atom_count += 1
                continue

            # Check 3: Atom type conservation (product types subset of reactant types)
            product_types = get_atom_types(prod_smiles)
            if not product_types.issubset(reactant_types):
                skipped_atom_type += 1
                continue

            # Check 4: Charge conservation
            product_charge = get_formal_charge(prod_smiles)
            if product_charge != reactant_charge:
                skipped_charge += 1
                continue

            # All checks passed, add to list
            new_reaction_records.append({
                'reactant1_smiles': norm_pair[0],
                'reactant2_smiles': norm_pair[1],
                'product_smiles': prod_smiles
            })
            generated_count += 1

            if generated_count > 0 and generated_count % 10000 == 0:
                print(f"... Generated {generated_count} new candidates ...")

        if (i + 1) % 50 == 0:
            progress = (i + 1) / len(reactant_combos) * 100
            print(f"Reactant combo progress: {progress:.1f}% ({i + 1}/{len(reactant_combos)})")

    # 5. Save Results
    if not new_reaction_records:
        print("Generation complete. No new valid candidates were found.")
        return

    output_df = pd.DataFrame(new_reaction_records)
    output_df.to_csv(output_csv, index=False)

    print("\n--- Generation Complete ---")
    print(f"Total new candidates generated: {generated_count}")
    print(f"Skipped (already existing): {skipped_existing}")
    print(f"Skipped (atom count mismatch): {skipped_atom_count}")
    print(f"Skipped (atom type mismatch): {skipped_atom_type}")
    print(f"Skipped (charge mismatch): {skipped_charge}")
    print(f"Results saved to: {output_csv}")


if __name__ == '__main__':
    generate_candidates_from_pools(
        original_data_csv='data/suzuki_miyaura_smiles_with_fingerprint.csv',
        reactants_pool_csv='data/reactants_pool.csv',
        products_pool_csv='data/products_pool.csv',
        output_csv='data/candidate_reactions_from_pool.csv'
    )