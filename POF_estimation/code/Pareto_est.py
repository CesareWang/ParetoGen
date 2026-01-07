import argparse
import os
import re
import glob
import pandas as pd
import numpy as np
from pygmo import hypervolume

OBJECTIVE_COLUMNS = ["CMC", "SFTcmc", "BCF", "biodeg", "PPARA", "PPARG"]
OPTIMIZATION_SENSES = ["min", "min", "min", "min", "min", "min"]
REFERENCE_POINT = [1, 30, 5, 0, 50, 50]
POF_FILTER_CONDITIONS = {
    'CMC': 1, 'SFTcmc': 30, 'BCF': 5,
    'biodeg': 0, 'PPARA': 50, 'PPARG': 50
}
INTERMEDIATE_PREFIX = "intermediate_pof_"

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Process a single PFAS data file to compute its Pareto Optimal Front (POF), "
                    "then merges all existing intermediate POFs to create a combined result."
    )
    parser.add_argument('--input-file', type=str, required=True, 
                        help="Name of the input CSV file (e.g., 'my_data_run_1.csv').")
    parser.add_argument('--input-dir', type=str, required=True, 
                        help="Directory containing the input data file.")
    parser.add_argument('--output-dir', type=str, required=True, 
                        help="Directory to save intermediate and final result files.")
    return parser.parse_args()

def find_pareto_front_mask(points, senses):
    num_points = points.shape[0]
    is_optimal = np.ones(num_points, dtype=bool)
    
    multipliers = np.array([-1 if s == 'max' else 1 for s in senses])
    processed_points = points * multipliers

    for i in range(num_points):
        if not is_optimal[i]:
            continue
        
        for j in range(num_points):
            if i == j:
                continue
            
            if np.all(processed_points[j] <= processed_points[i]) and np.any(processed_points[j] < processed_points[i]):
                is_optimal[i] = False
                break
    
    return is_optimal

def contains_pfas(smiles):
    pattern = r'C(\(F\)){2,}'
    return bool(re.search(pattern, str(smiles)))

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    print(f"\nProcessing {os.path.basename(filepath)}...")
    print(f"Initial raw data count: {len(df)}")
    
    df.dropna(subset=['cano_smiles'], inplace=True)
    df.drop_duplicates(subset=['cano_smiles'], inplace=True)
    print(f"After removing duplicates and NaN smiles: {len(df)}")
    
    mask = df['cano_smiles'].apply(contains_pfas)
    df_filtered = df[mask].copy()
    print(f"After filtering for PFAS structures: {len(df_filtered)}")
    
    df_filtered['biodeg'] = -df_filtered['biodeg']
    df_filtered['PPARA'] = -df_filtered['PPARA']
    df_filtered['PPARG'] = -df_filtered['PPARG']
    
    return df_filtered

def apply_pof_filters(df, conditions):
    query_str = " and ".join([f"`{col}` < {val}" for col, val in conditions.items()])
    return df.query(query_str)

def calculate_pareto_front(df, columns, senses):
    if df.empty:
        return df
    objectives = df[columns].values
    mask = find_pareto_front_mask(objectives, senses)
    return df[mask]

def calculate_hypervolume(pof_df, columns, ref_point):
    if pof_df.empty:
        return 0.0
    points = pof_df[columns].values
    hv = hypervolume(points)
    return hv.compute(ref_point)

def main():
    args = setup_arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_filepath = os.path.join(args.input_dir, args.input_file)
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        return

    data = load_and_prepare_data(input_filepath)
    pof_individual = calculate_pareto_front(data, OBJECTIVE_COLUMNS, OPTIMIZATION_SENSES)
    pof_individual_filtered = apply_pof_filters(pof_individual, POF_FILTER_CONDITIONS)
    
    print(f"Individual Pareto front size: {len(pof_individual_filtered)}")
    hv_individual = calculate_hypervolume(pof_individual_filtered, OBJECTIVE_COLUMNS, REFERENCE_POINT)
    print(f"Individual hypervolume: {hv_individual}")
    
    base_name = os.path.basename(args.input_file)
    intermediate_pof_filename = f"{INTERMEDIATE_PREFIX}{base_name}"
    intermediate_pof_filepath = os.path.join(args.output_dir, intermediate_pof_filename)
    pof_individual_filtered.to_csv(intermediate_pof_filepath, index=False)
    print(f"Intermediate POF saved to: {intermediate_pof_filepath}")

    print("\n--- Merging and Processing All Available Intermediate Data ---")
    
    search_pattern = os.path.join(args.output_dir, f"{INTERMEDIATE_PREFIX}*.csv")
    intermediate_files = glob.glob(search_pattern)
    
    if not intermediate_files:
        print("No intermediate POF files found to merge. Exiting merge step.")
        return

    print(f"Found {len(intermediate_files)} intermediate files to merge.")
    all_pofs = [pd.read_csv(f) for f in intermediate_files]
    
    combined_df = pd.concat(all_pofs, ignore_index=True)
    combined_df.drop_duplicates(subset='cano_smiles', inplace=True)
    print(f"Total unique entries after merging: {len(combined_df)}")

    pof_combined = calculate_pareto_front(combined_df, OBJECTIVE_COLUMNS, OPTIMIZATION_SENSES)
    print(f"Combined Pareto front size: {len(pof_combined)}")
    
    hv_combined = calculate_hypervolume(pof_combined, OBJECTIVE_COLUMNS, REFERENCE_POINT)
    print(f"Combined hypervolume: {hv_combined}")

    final_output_filename = "combined_unique_POF.csv"
    final_output_filepath = os.path.join(args.output_dir, final_output_filename)
    pof_combined.to_csv(final_output_filepath, index=False)
    print(f"Final combined POF saved to: {final_output_filepath}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()