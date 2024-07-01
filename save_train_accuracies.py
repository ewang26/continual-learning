# this file is for the purpose of saving and pushing the accuracies of trainining runs from cluster. 
# we can delete this file later.

import numpy as np
import os

import yaml
import argparse
from tqdm import tqdm


def save_downstream_acc(p_vals, dataset_name, memory_method_arr, num_tasks):
    num_p = len(p_vals)

    grad_sim_dir = 'gradient_similarity'
    if not os.path.exists(grad_sim_dir): os.mkdir(grad_sim_dir)

    dataset_save_dir = f'{grad_sim_dir}/{dataset_name}'
    if not os.path.exists(dataset_save_dir): os.mkdir(dataset_save_dir)

    # loop through memory methods
    for memory_method in memory_method_arr:

        mem_save_dir = f'{dataset_save_dir}/{memory_method}'
        if not os.path.exists(mem_save_dir): os.mkdir(mem_save_dir)
        result_file_path = f'{mem_save_dir}/acc_block.npy'

        # we want to store all 5 task downstream acc for all p for this memory method
        acc_block = np.zeros((num_p, num_tasks))
        for p_index, p in enumerate(p_vals):
            acc_block[p_index] = np.load(f'models/{dataset_name}/{memory_method}/{p}/train_0/acc.npy')

        np.save(result_file_path, acc_block)
        

def main(cd):

    p_vals = cd['p_vals']
    model_weight_types = cd['model_weight_types']
    model_layer_names = cd['model_layer_names']
    dataset_name = cd['dataset_name']
    grad_type_arr = cd['grad_type_arr']
    memory_method_arr = cd['memory_method_arr']
    num_tasks = cd['num_tasks']
    num_ideal_models = cd['num_ideal_models']
    num_runs = cd['num_runs']

    save_downstream_acc(p_vals = p_vals,
                        dataset_name = dataset_name,
                        memory_method_arr = memory_method_arr,
                        num_tasks = num_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file to run from.",
    )
    args = parser.parse_args()

    with open(f"{args.config}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(config_dict)

    main(config_dict)
    print("comparison done!")




















