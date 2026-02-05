import pandas as pd
import os


def build_fragment_pools(input_csv: str,
                         reactants_output_csv: str,
                         products_output_csv: str,
                         yield_threshold: float = 60.0,
                         top_n_reactants: int = 300,
                         top_n_products: int = 300):
    """
    Step 1: Build reactant and product fragment pools from high-yield reactions.
    MODIFIED: Merges with existing pools and saves ONLY the SMILES column.
    """
    print("--- Step 1: Starting to build fragment pools ---")

    # Ensure output directory exists
    output_dir = os.path.dirname(reactants_output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_csv}")
        return

    required_cols = ['reactant1_smiles', 'reactant2_smiles', 'product_smiles', 'y_val']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain columns: {required_cols}")
        return

    print(f"Successfully loaded {len(df)} original reactions.")

    # 1. Filter for high-yield reactions
    df_high_yield = df[df['y_val'] >= yield_threshold].copy()
    if len(df_high_yield) == 0:
        print(f"Warning: No reactions found with yield >= {yield_threshold}. Try lowering the threshold.")
        return
    print(f"Filtered {len(df_high_yield)} reactions with yield >= {yield_threshold}.")

    # 2. Build reactant pool
    reactants_series = pd.concat([
        df_high_yield['reactant1_smiles'],
        df_high_yield['reactant2_smiles']
    ]).dropna().astype(str)
    new_reactants_counts_df = reactants_series.value_counts().reset_index()
    new_reactants_counts_df.columns = ['smiles', 'count']

    # --- Load and merge existing pool ---
    if os.path.exists(reactants_output_csv):
        print(f"Found existing reactant pool. Merging... {reactants_output_csv}")
        try:
            # MODIFIED: Load existing pool, which now only has 'smiles'
            existing_reactants_df = pd.read_csv(reactants_output_csv)
            # Give existing smiles a "count" of 0 for aggregation,
            # new counts will be added. This assumes we only care about new counts ranking.
            # A better way: Load, merge, *then* count.

            # --- REVISED MERGE LOGIC ---
            # To correctly merge, the existing pool needs its 'counts'.
            # If we only save 'smiles', we lose the count history.
            #
            # Assumption: The user wants to merge the *SMILES lists* not the counts.
            # Let's load the old SMILES, get the new SMILES, combine, and get unique top N.
            #
            # Re-reading the previous script... we were merging counts.
            # If we only save SMILES, we *cannot* merge counts.

            # --- New Interpretation ---
            # Let's assume the existing CSVs *with counts* are needed for the logic,
            # but the *final output* should just be smiles.
            #
            # The previous script *loaded* CSVs assuming 'smiles' and 'count' existed.
            # If we save only 'smiles', the *next* run will fail.
            #
            # Let's stick to the previous logic (merging counts)
            # but ONLY save the 'smiles' column at the very end.

            # Load existing pool (assuming it has 'smiles' and 'count')
            existing_reactants_df = pd.read_csv(reactants_output_csv)
            if 'count' not in existing_reactants_df.columns:
                print(f"Warning: Existing pool {reactants_output_csv} has no 'count'. Resetting pool.")
                agg_reactants_df = new_reactants_counts_df
            else:
                combined_reactants_df = pd.concat([existing_reactants_df, new_reactants_counts_df])
                agg_reactants_df = combined_reactants_df.groupby('smiles')['count'].sum().reset_index()

        except (pd.errors.EmptyDataError, FileNotFoundError):
            print(f"Warning: Existing pool {reactants_output_csv} is empty or unreadable. Using new data only.")
            agg_reactants_df = new_reactants_counts_df
        except KeyError:
            print(f"Warning: Existing pool {reactants_output_csv} is in the old format (smiles only). Resetting pool.")
            agg_reactants_df = new_reactants_counts_df
    else:
        print("No existing reactant pool found. Creating new one.")
        agg_reactants_df = new_reactants_counts_df

    top_reactants_df = agg_reactants_df.sort_values(by='count', ascending=False).head(top_n_reactants)

    # 3. Build product pool
    products_series = df_high_yield['product_smiles'].dropna().astype(str)
    new_products_counts_df = products_series.value_counts().reset_index()
    new_products_counts_df.columns = ['smiles', 'count']

    # --- Load and merge existing pool ---
    if os.path.exists(products_output_csv):
        print(f"Found existing product pool. Merging... {products_output_csv}")
        try:
            existing_products_df = pd.read_csv(products_output_csv)
            if 'count' not in existing_products_df.columns:
                print(f"Warning: Existing pool {products_output_csv} has no 'count'. Resetting pool.")
                agg_products_df = new_products_counts_df
            else:
                combined_products_df = pd.concat([existing_products_df, new_products_counts_df])
                agg_products_df = combined_products_df.groupby('smiles')['count'].sum().reset_index()

        except (pd.errors.EmptyDataError, FileNotFoundError):
            print(f"Warning: Existing pool {products_output_csv} is empty or unreadable. Using new data only.")
            agg_products_df = new_products_counts_df
        except KeyError:
            print(f"Warning: Existing pool {products_output_csv} is in the old format (smiles only). Resetting pool.")
            agg_products_df = new_products_counts_df
    else:
        print("No existing product pool found. Creating new one.")
        agg_products_df = new_products_counts_df

    top_products_df = agg_products_df.sort_values(by='count', ascending=False).head(top_n_products)

    if top_reactants_df.empty or top_products_df.empty:
        print("Error: Failed to build one or both pools (they might be empty).")
        return

    print(f"Identified Top-{len(top_reactants_df)} reactants and Top-{len(top_products_df)} products (after merging).")

    # 4. Save pools to files (ONLY SMILES column)

    # --- MODIFICATION HERE ---
    # Select only the 'smiles' column before saving
    top_reactants_df_smiles_only = top_reactants_df[['smiles']]
    top_products_df_smiles_only = top_products_df[['smiles']]

    top_reactants_df_smiles_only.to_csv(reactants_output_csv, index=False)
    print(f"Reactant pool (SMILES only) saved to: {reactants_output_csv}")

    top_products_df_smiles_only.to_csv(products_output_csv, index=False)
    print(f"Product pool (SMILES only) saved to: {products_output_csv}")

    print("--- Fragment pool building complete ---")


if __name__ == '__main__':
    build_fragment_pools(
        input_csv='data/train_processed_can.csv',
        reactants_output_csv='data/reactants_pool.csv',
        products_output_csv='data/products_pool.csv',
        yield_threshold=10,
        top_n_reactants=250,
        top_n_products=250
    )