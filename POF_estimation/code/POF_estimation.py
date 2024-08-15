import numpy as np
import pandas as pd
import importlib
import collections.abc
import numba

# Check if a package is available
def user_has_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ModuleNotFoundError:
        return False

# Validate inputs and convert to a consistent format
def validate_inputs(costs, sense=None):
    if isinstance(costs, np.ndarray):
        if costs.ndim == 1:
            return validate_inputs(costs.copy().reshape(-1, 1), sense=sense)
        if costs.ndim != 2:
            raise ValueError("`costs` must have shape (observations, objectives).")
        costs = costs.copy()
    elif user_has_package("pandas") and not isinstance(costs, pd.DataFrame):
        return validate_inputs(np.asarray(costs), sense=sense)
    else:
        return validate_inputs(np.asarray(costs), sense=sense)
    
    n_costs, n_objectives = costs.shape
    sense = list(sense or ["min"] * n_objectives)

    if not isinstance(sense, collections.abc.Sequence) or len(sense) != n_objectives:
        raise ValueError("`sense` must be a sequence matching the number of objectives.")

    sense_map = {"min": "min", "minimum": "min", "max": "max", "maximum": "max", 
                 "diff": "diff", "different": "diff"}
    sense = [sense_map.get(s.lower()) for s in sense]

    if not all(s in ["min", "max", "diff"] for s in sense):
        raise TypeError("`sense` must be one of: ['min', 'max', 'diff']")

    return costs, sense

@numba.jit(nopython=True)
def dominates(a, b, length):
    better = False
    for i in range(length):
        if a[i] > b[i]:
            return False
        if a[i] < b[i]:
            better = True
    return better

@numba.jit(nopython=True)
def BNL(costs, distinct=True):
    n_costs, n_objectives = costs.shape
    is_efficient = np.arange(n_costs)
    num_efficient = 1
    window_changed = True

    for i in range(1, n_costs):
        this_cost = costs[i]
        if window_changed:
            window = costs[is_efficient[:num_efficient]]
            window_rows, window_cols = window.shape
            window_changed = False

        if window_dominates_cost(window, this_cost, window_rows, window_cols) >= 0:
            continue

        dominated_inds_window = cost_dominates_window(window, this_cost, window_rows, window_cols)
        if dominated_inds_window:
            for to_remove in [is_efficient[k] for k in dominated_inds_window]:
                for j, efficient in enumerate(is_efficient):
                    if efficient == to_remove:
                        is_efficient[j:num_efficient] = is_efficient[j + 1:num_efficient + 1]
                        num_efficient -= 1
                        break

        if not distinct or not any_equal_jitted(window, this_cost):
            is_efficient[num_efficient] = i
            num_efficient += 1
            window_changed = True

    bools = np.zeros(n_costs, dtype=np.bool_)
    bools[is_efficient[:num_efficient]] = 1
    return bools

def paretoset(costs, sense=None, distinct=True, use_numba=True):
    paretoset_algorithm = BNL if user_has_package("numba") and use_numba else paretoset_efficient
    costs, sense = validate_inputs(costs, sense)
    n_costs, n_objectives = costs.shape

    max_cols = [i for i in range(n_objectives) if sense[i] == "max"]
    min_cols = [i for i in range(n_objectives) if sense[i] == "min"]

    if isinstance(costs, pd.DataFrame):
        costs = costs.to_numpy(copy=True)
    for col in max_cols:
        costs[:, col] = -costs[:, col]

    return paretoset_algorithm(costs, distinct=distinct)

def estimate_pof():
    df = pd.read_csv('..\data\POF_data.csv')
    objectives = df[['CMC', 'ST', 'KOW']]
    mask = paretoset(objectives, sense=["min", "min", "min"])
    paretosets = df[mask]

    paretosets_list = {
        'CMC': paretosets['CMC'],
        'ST': paretosets['ST'],
        'KOW': paretosets['KOW'],
        'tox': paretosets['tox_pred_label_sum'],
        'cano_smiles': paretosets['cano_smiles']
    }
    paretosets_df = pd.DataFrame(paretosets_list)
    paretosets_df.to_csv('../results/new_POF_df.csv', index=False)

if __name__ == "__main__":
    estimate_pof()