import numpy as np
import os

import yaml
import argparse
from tqdm import tqdm


# converting to square

# MSE
def mse(actual, ideal):
    return np.mean((actual.flatten() - ideal.flatten()) ** 2)

# Cosine Similarity
def cos_sim(actual, ideal):
    actual_flat, ideal_flat = actual.flatten(), ideal.flatten()
    dot = np.dot(actual_flat, ideal_flat)
    if (np.linalg.norm(actual_flat) == 0) or (np.linalg.norm(ideal_flat) == 0):
        return 0
    else:
        return dot / (np.linalg.norm(actual_flat) * np.linalg.norm(ideal_flat)) 

def compare_gradients(approx, ideal, metric_func):
    return metric_func(approx, ideal)

def get_grad_dist(metric_func, 
                p_vals, 
                model_weight_types, 
                model_layer_names,
                exp_name, 
                grad_type,
                memory_method, 
                num_tasks,
                num_ideal_models,
                num_runs):
    num_p = len(p_vals)

    num_grad_files = len(model_weight_types)*len(model_layer_names)

    if grad_type == 'past': # dont include task 0 
        data_tensor = np.zeros((num_p, num_runs, num_ideal_models, num_tasks-1, num_grad_files))
        task_idx_arr = np.arange(1, 5)
    else:
        data_tensor = np.zeros((num_p, num_runs, num_ideal_models, num_tasks, num_grad_files))
        task_idx_arr = np.arange(5)

    for p_index, p in enumerate(tqdm(p_vals, desc="Processing p values")):
        for run in range(num_runs):
            for ideal_index in range(num_ideal_models):
                for task_idx, task_val in enumerate(task_idx_arr):
                    grad_index = 0
                    for model_layer_prefix in model_layer_names: # loop over model layer names
                        for weight_type in model_weight_types: # loop over types of weights
                            weight_name = f'{model_layer_prefix}.{weight_type}.npy'
                            
                            p_grad_actual_arr = np.load(f'models/{exp_name}/{memory_method}/{p}/run_{run}/train_{ideal_index}/grad_task_{task_val}/{grad_type}_grad/{weight_name}')
                            p_grad_ideal_arr =  np.load(f'models/{exp_name}/random/1/run_0/train_{ideal_index}/grad_task_{task_val}/{grad_type}_grad/{weight_name}')

                            data_tensor[p_index, run, ideal_index, task_idx, grad_index] = compare_gradients(p_grad_actual_arr, p_grad_ideal_arr, metric_func)
                            grad_index += 1

    return data_tensor


def compute_gradient_similarity(metric_list, 
                                metric_names,
                                p_vals, 
                                model_weight_types, 
                                model_layer_names,
                                dataset_name, 
                                grad_type_arr,
                                memory_method_arr, 
                                num_tasks,
                                num_ideal_models,
                                num_runs):
    
    grad_sim_dir = 'gradient_similarity'
    # if not os.path.exists(grad_sim_dir): os.mkdir(grad_sim_dir)
    os.makedirs(grad_sim_dir, exist_ok=True)

    dataset_save_dir = f'{grad_sim_dir}/{dataset_name}'
    # if not os.path.exists(dataset_save_dir): os.mkdir(dataset_save_dir)
    os.makedirs(dataset_save_dir, exist_ok=True)

    # loop through memory methods
    for memory_method in memory_method_arr:

        mem_save_dir = f'{dataset_save_dir}/{memory_method}'
        #if not os.path.exists(mem_save_dir): os.mkdir(mem_save_dir)
        os.makedirs(mem_save_dir, exist_ok=True) #changed this

        # loop through metric functions
        for metric_index, metric_name in enumerate(metric_names):
            metric = metric_list[metric_index] # metric function
            metric_save_dir = f'{mem_save_dir}/{metric_name}'
            if not os.path.exists(metric_save_dir): os.mkdir(metric_save_dir)

            # loop through gradient types
            for grad_type in tqdm(grad_type_arr, desc="Processing gradient types"):
                result_file_path = f'{metric_save_dir}/{grad_type}_gradient_comp.npy'

                data_block = get_grad_dist(metric_func = metric, 
                                           p_vals = p_vals, 
                                           model_weight_types = model_weight_types, 
                                           model_layer_names = model_layer_names,
                                           exp_name = dataset_name, 
                                           grad_type = grad_type,
                                           memory_method = memory_method, 
                                           num_tasks = num_tasks,
                                           num_ideal_models = num_ideal_models,
                                           num_runs = num_runs)

                np.save(result_file_path, data_block)

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
    metric_names = cd['metric_names']
    if len(metric_names) == 1 and metric_names[0] == 'Cosine Similarity':
        metric_list = [cos_sim]
    else:
        print('other metrics not implemented')
        assert False

    p_vals = cd['p_vals']
    model_weight_types = cd['model_weight_types']
    model_layer_names = cd['model_layer_names']
    dataset_name = cd['dataset_name']
    grad_type_arr = cd['grad_type_arr']
    memory_method_arr = cd['memory_method_arr']
    num_tasks = cd['num_tasks']
    num_ideal_models = cd['num_ideal_models']
    num_runs = cd['num_runs']

    compute_gradient_similarity(metric_list = metric_list, 
                               metric_names = metric_names,
                               p_vals = p_vals, 
                               model_weight_types = model_weight_types, 
                               model_layer_names = model_layer_names,
                               dataset_name = dataset_name, 
                               grad_type_arr = grad_type_arr,
                               memory_method_arr = memory_method_arr, 
                               num_tasks = num_tasks,
                               num_ideal_models = num_ideal_models,
                               num_runs = num_runs)
    
    # currently only saves acc for 1 downstream trained model (in the future we can store statistics on multiple models)
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




















