import os
import sys
import pickle
import argparse
import time
import gc

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from AttentiveFP import save_smiles_dicts, get_smiles_array

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True
torch.nn.Module.dump_patches = True
sys.setrecursionlimit(50000)

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Molecular Property Prediction')
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model-details', type=str, required=True)
    return parser.parse_args()

def preprocess_data(input_path, output_dir):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    preprocessed_path = os.path.join(output_dir, f"preprocessed_{base_name}.csv")
    feature_path = os.path.join(output_dir, f"features_{base_name}.pickle")

    print(f"--- Starting Preprocessing for {os.path.basename(input_path)} ---")
    smiles_tasks_df = pd.read_csv(input_path)
    smilesList = smiles_tasks_df.smiles.values
    print(f"Original number of SMILES: {len(smilesList)}")

    remained_smiles, canonical_smiles_list = [], []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                remained_smiles.append(smiles)
        except Exception:
            pass
    
    print(f"Successfully processed SMILES: {len(remained_smiles)}")
    processed_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)].copy()
    processed_df['cano_smiles'] = canonical_smiles_list
    
    valid_smiles = [s for s in canonical_smiles_list if Chem.MolFromSmiles(s) and len(Chem.MolFromSmiles(s).GetAtoms()) < 200]
    
    if os.path.isfile(feature_path):
        print(f"Loading existing features from: {feature_path}")
        with open(feature_path, "rb") as f:
            feature_dicts = pickle.load(f)
    else:
        print(f"Generating and saving features to: {feature_path}")
        feature_dicts = save_smiles_dicts(valid_smiles, feature_path.replace('.pickle', ''))
    
    final_df = processed_df[processed_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    final_df = final_df.drop_duplicates(subset=['cano_smiles'], keep='first')
    
    final_df.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to: {preprocessed_path}")
    print(f"--- Preprocessing Complete ---")
    
    return final_df, feature_dicts

def run_predictions(data_df, feature_dicts, models_dir, output_dir, device, model_detail_files):
    print("\n--- Starting Model Predictions ---")
    base_name = f"predictions_for_{os.path.splitext(os.path.basename(args.input_file))[0]}"
    
    results_df = data_df[['smiles', 'cano_smiles']].copy()

    for model_file in model_detail_files:
        model_name = model_file.split('_')[2].split('.')[0]
        print(f"\nProcessing property: {model_name}")
        model_details_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_details_path):
            print(f"Warning: Model details file not found, skipping: {model_details_path}")
            continue
            
        model_details_df = pd.read_csv(model_details_path, index_col='index')
        
        for model_id in tqdm(model_details_df.index, desc=f"Predicting {model_name}"):
            model_path = os.path.join(models_dir, model_details_df.loc[model_id, 'model_path'])
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found, skipping: {model_path}")
                continue

            model = torch.load(model_path, map_location=device)
            model.to(device)
            model.eval()

            batch_size = 32
            y_pred_label = []
            
            for i in range(0, data_df.shape[0], batch_size):
                batch_df = data_df.iloc[i:i+batch_size]
                smiles_list = batch_df.cano_smiles.values
                
                x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array(smiles_list, feature_dicts)
                
                with torch.no_grad():
                    _, mol_prediction = model(
                        torch.Tensor(x_atom).to(device),
                        torch.Tensor(x_bonds).to(device),
                        torch.LongTensor(x_atom_index).to(device),
                        torch.LongTensor(x_bond_index).to(device),
                        torch.Tensor(x_mask).to(device)
                    )
                y_pred_label.extend(mol_prediction.cpu().numpy().flatten().tolist())
            
            results_df[model_name] = y_pred_label

    final_output_path = os.path.join(output_dir, f"labeled_all_properties_{base_name}.csv")
    results_df.to_csv(final_output_path, index=False)
    print(f"\n--- All Predictions Complete ---")
    print(f"Final combined results saved to: {final_output_path}")

def main(args):
    torch.manual_seed(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    input_filepath = os.path.join(args.data_dir, args.input_file)

    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        return

    preprocessed_df, feature_dicts = preprocess_data(input_filepath, args.output_dir)
    
    if preprocessed_df.empty:
        print("No valid molecules found after preprocessing. Exiting.")
        return
        
    run_predictions(preprocessed_df, feature_dicts, args.models_dir, args.output_dir, device, args.model_detail_files)

if __name__ == '__main__':
    args = setup_arg_parser()
    main(args)