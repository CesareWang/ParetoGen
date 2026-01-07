import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run novo analysis')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)

if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = True
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    repeat = args['repeat']
    
    # get back the experiment parameters
    mode = config['EXPERIMENTS']['mode']
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    aug = int(config['AUGMENTATION']['fold'])
    dir_data = str(config['DATA']['dir'])
    name_data = str(config['DATA']['name'])
    
    # experiment parameters depending on the mode
    aug = int(config['AUGMENTATION']['fold'])
    
    if verbose: print('\nSTART NOVO ANALYSIS')
    ####################################
   
    ####################################
    # Path to save the novo analysis and to the generated data
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/novo_molecules/{repeat}/'
        path_gen = f'{dir_exp}/{mode}/{exp_name}/generated_data/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/novo_molecules/'
        path_gen = f'{dir_exp}/{mode}/{exp_name}/generated_data/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################    
    
    ####################################
    # Load the fine-tuning data to compare the novelty of the generated SMILES strings
    # 不再加载预训练数据，只使用微调数据
    
    # fine-tuning data
    dir_data_ft = str(config['DATA']['dir'])
    name_data_ft = str(config['DATA']['name'])
    full_path_ft = f'{dir_data_ft}{name_data_ft}/{min_len}_{max_len}_x{aug}/'
    with open(f'{full_path_ft}data_tr.txt', 'r') as f:
        ft_data_training = f.readlines()
    with open(f'{full_path_ft}data_val.txt', 'r') as f:
        ft_data_validation = f.readlines()
    full_ft_data = ft_data_training + ft_data_validation
    
    # Use only fine-tuning data for novelty comparison
    reference_data = list(set(full_ft_data))
    ####################################
    
    ####################################
    # Initialize lists to collect all molecules
    all_generated_molecules = []  # 所有生成的分子（原始）
    all_valid_molecules = []      # 所有有效分子（规范化后）
    all_novo_molecules = []       # 所有新颖分子（相对于微调数据）
    
    # Start iterating over the files
    t0 = time.time()
    for filename in os.listdir(path_gen):
        if filename.endswith('.pkl'):
            name = filename.replace('.pkl', '')
            data = hp.load_obj(path_gen + name)
                        
            valids = []
            n_valid = 0
            
            for gen_smile in data:
                if len(gen_smile)!=0 and isinstance(gen_smile, str):
                    # 添加到所有生成分子列表
                    all_generated_molecules.append(gen_smile)
                    
                    gen_smile = gen_smile.replace(pad_char,'')
                    gen_smile = gen_smile.replace(end_char,'')
                    gen_smile = gen_smile.replace(start_char,'')
                    
                    mol = Chem.MolFromSmiles(gen_smile)
                    if mol is not None: 
                        cans = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                        if len(cans)>=1:
                            n_valid+=1
                            valids.append(cans)
                            # 添加到所有有效分子列表
                            all_valid_molecules.append(cans)
                    
            if n_valid!=0:
                # Now let's pruned our valid guys
                unique_set = set(valids)
                n_unique = len(unique_set)
                
                # 只与微调数据比较新颖性
                novo_ft = list(unique_set - set(reference_data))
                n_novo_ft = len(novo_ft)
                
                novo_analysis = {'n_valid': n_valid,
                                 'n_unique': n_unique,
                                 'n_novo_ft': n_novo_ft,
                                 'novo_ft': novo_ft}
                
                # we save the novo molecules also as .txt
                novo_name = f'{save_path}molecules_{name}'
                with open(f'{novo_name}.txt', 'w+') as f:
                    for item in novo_ft:
                        f.write("%s\n" % item)
                        
                hp.save_obj(novo_analysis, novo_name)
                
                if verbose: print(f'sampling analysis for {name} done')
            else:
                print(f'There are n {n_valid} valids SMILES for {name}')
    
    ####################################
    # Calculate overall statistics and save combined files
    if all_generated_molecules:
        # 去重处理
        all_valid_unique = list(set(all_valid_molecules))
        all_novo_unique = list(set(all_valid_unique) - set(reference_data))
        
        # 计算总体统计
        total_generated = len(all_generated_molecules)
        total_valid = len(all_valid_molecules)
        total_valid_unique = len(all_valid_unique)
        total_novo = len(all_novo_unique)
        
        validity_rate = (total_valid / total_generated) * 100 if total_generated > 0 else 0
        novelty_rate = (total_novo / total_valid_unique) * 100 if total_valid_unique > 0 else 0
        
        # 保存所有生成的分子到一个文件
        with open(f'{save_path}all_generated_molecules.txt', 'w+') as f:
            for molecule in all_generated_molecules:
                f.write("%s\n" % molecule)
        
        # 保存所有有效分子到一个文件
        with open(f'{save_path}all_valid_molecules.txt', 'w+') as f:
            for molecule in all_valid_unique:
                f.write("%s\n" % molecule)
        
        # 保存所有新颖分子到一个文件（相对于微调数据）
        with open(f'{save_path}all_novo_molecules.txt', 'w+') as f:
            for molecule in all_novo_unique:
                f.write("%s\n" % molecule)
        
        # 保存总体统计信息
        overall_stats = {
            'total_generated': total_generated,
            'total_valid': total_valid,
            'total_valid_unique': total_valid_unique,
            'total_novo': total_novo,
            'validity_rate': validity_rate,
            'novelty_rate': novelty_rate,
            'reference_data_size': len(reference_data)
        }
        
        # 保存为pickle文件
        hp.save_obj(overall_stats, f'{save_path}overall_statistics')
        
        # 同时保存为可读的txt文件
        with open(f'{save_path}overall_statistics.txt', 'w+') as f:
            f.write("Overall Generation Statistics\n")
            f.write("=============================\n\n")
            f.write(f"Reference data size (fine-tuning): {len(reference_data)}\n")
            f.write(f"Total generated molecules: {total_generated}\n")
            f.write(f"Total valid molecules: {total_valid}\n")
            f.write(f"Total unique valid molecules: {total_valid_unique}\n")
            f.write(f"Total novel molecules (vs fine-tuning): {total_novo}\n")
            f.write(f"Validity rate: {validity_rate:.2f}%\n")
            f.write(f"Novelty rate (vs fine-tuning): {novelty_rate:.2f}%\n")
        
        if verbose:
            print(f"\nOVERALL STATISTICS:")
            print(f"Reference data size (fine-tuning): {len(reference_data)}")
            print(f"Total generated: {total_generated}")
            print(f"Total valid: {total_valid} ({validity_rate:.2f}%)")
            print(f"Total unique valid: {total_valid_unique}")
            print(f"Total novel (vs fine-tuning): {total_novo} ({novelty_rate:.2f}%)")
    
    end = time.time()
    if verbose: print(f'NOVO ANALYSIS DONE in {end - start:.04} seconds')
    ####################################