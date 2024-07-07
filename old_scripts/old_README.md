# Continual Learning Starter Code

This repository contains code for running basic continual learning training pipelines. 

## Getting Started

### Environment Setup

Download packages from `requirements.txt` using: 

```
pip install -r requirements.txt
```

### Running a Pipeline

Below is a step-by-step example of how to train the full pipeline, using the MNIST dataset:

[1] First, we will train ideal models by using the full dataset. This will store the weights that we use to evaluate the loss afterwards. Note the following value in `mnist_train_ideal_model.yaml`:
* `p_arr: [1]`
* `num_samples: [10]`
```
main_batch.py --config configs/mnist_train_ideal_model.yaml
```

[2] Because p = 1, it doesnt matter which memory selection method we use to train these models. For this example, we used random selection. Now, we will move these models into the `ideal_model/` folder, after which we will not modify. These weights will be used for all gradient evaluation afterwards.

```
cp -r models/mnist_split/random/1 models/mnist_split/ideal_model
```

[3] Now that we have our ideal weights, we can begin to compare gradients. For each memory selection method, we sample multiple memory sets, and for each ideal model weight, we evaluate the gradients for each task. In this example, we grab the configs for Random, Reservoir Sampling, GSS, Lambda, and Kmeans. Note the current values in the config files:
* `p_arr: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]`
* `num_samples: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]`
* `num_ideal_models: 10`

```
python main_batch.py --config configs/random/config_mnist_grad_eval.yaml
python main_batch.py --config configs/class_balanced/config_mnist_grad_eval.yaml
python main_batch.py --config configs/GSS/config_mnist_grad_eval.yaml
python main_batch.py --config configs/lambda/config_mnist_grad_eval.yaml
python main_batch.py --config configs/kmeans/config_mnist_grad_eval.yaml
```

[4] This step can be done before or after step 3. Now, we will train downstream models using memory sets instead of the full dataset. For each memory selection method, we train with the train config file, but now with different p-values. In this example, we train 1 downstream model for each p-value.
* `p_arr: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]`
* `num_samples: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

```
python main_batch.py --config configs/random/config_mnist_train.yaml
python main_batch.py --config configs/class_balanced/config_mnist_train.yaml
python main_batch.py --config configs/GSS/config_mnist_train.yaml
python main_batch.py --config configs/lambda/config_mnist_train.yaml
python main_batch.py --config configs/kmeans/config_mnist_train.yaml
```

[5] Now that all the gradient values are saved, we can compute similarities between the ideal gradients (full dataset) with the reconstructed gradients (memory dataset). Note that the similairty evaluation is split into 2 calls, one for random and one for the rest of the memory methods. This is just for debugging purposes so we can run a full random pipeline first.

```
python save_similarity_block_v2.py --config configs/grad_eval/mnist_random.yaml
python save_similarity_block_v2.py --config configs/grad_eval/mnist_rest.yaml
```

[6] All the gradient comparison data will be saved in `gradient_similarity/mnist_split/{memory_selection_method}/{metric_name}/`. The huge gradient data plot has the shape `(p_vals, num_runs, num_ideal_models, num_tasks, num_grad_layers)`. Specific for this example, the data block has shape `(10, 10, 10, 4, 4)`, since we compute past gradients for tasks 1-4, and there are 4 differnet gradient layers in the MLP.

[7] After the gradient comparison is saved, the data block can then be used for plotting any statistical averages. See `grad_similarity_plots_restructure.ipynb` for example plotting code. The plot generated from this notebook are saved in `gradient_similarity/mnist_split/plots/`

[8] For running on the cluster, I've provided two .job files that first perform the pipeline on random (including the ideal model training), and then running the pipeline on the rest of the memory selection methods:

```
sbatch mnist_random_pipeline.job
sbatch mnist_memory_train_and_eval.job
```

### Config Files

The code uses yaml config files. See an example in  `./configs/exampple_config.yaml`. 
We provide an description of the different config fields: 


* `use_wandb`: if True, then results logged to weights and biases.
* `wandb_project_name`: name of wandb project.
* `wandb_profile`: wandb user profile. 
* `memory_set_manager`: describes the way that memory sets of previous tasks are selected.
* `p`: If using memory set manager, the probability of including a point from the 
task dataset in the memory set. TODO make this generic memory set manager prarams.
* `use_memory_set`: If False, then previous tasks use entire datasets. Else the memory 
set manager is used to select memory subsets.
* `learning_manager`: manager for the continual learning training. Essentially managers 
what different tasks the continual learning training consists of.
* `lr`: learning rate.
* `batch_size`: training batch size.
* `random_seed`: random seed.
* `epochs`: number of training epochs per task.
* `model_save_dir`: directory to save trained models to after each task.
* `model_load_dir`: directory to load trained models from. If given, them 
model_save_dir is ignored and no model will actually be trained, only 
pretrained models are loaded from memory.
* `experiment_tag`: tag put on wandb run.
* `experiment_metadata_path`: path to a csv file that the wandb run ID will be written to. 
Useful for pulling the data and visualizing later.
