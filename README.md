### Code Base Structure

**Methods for Memory Selection and Continual Learning**
1. `data.py` contains definition of all memory set selection methods. 
    - each method is a `class` extending the base class `MemorySetManager`
    - `__init__` sets the percentage of points retained for each memory set, as well as method specific parameters
    - `create_memory_set` takes in the data, `X, y`, for a given task and returns the memory set, `memory_x, memory_y` for the task
        - memory sets of `RandomMemorySetManager`, `KMeansMemorySetManager`,`iCaRL` are created when `create_memory_set` is called, and do not require updates
        - for `LambdaMemorySetManager` and `GSSMemorySetManager` empty tensors are returned by `create_memory_set`. Instead, memory sets are updated via other class methods, and require access to gradients from a model		
2. `train_task.py` contains the model training, memory selection and evaluation pipeline.
    - given: train/test data for tasks $t = 1,\ldots, T$; instantiated pytorch models `M1`, `M2`, `M3`; instantiated memory set manager `memory_set_manager` and parameters for model training.
    - calling `CL_tasks` executes the following:
        - if the `use_memory_sets` flag is set to `False`, then we train models on full training datasets:
            - Train model `M1` on $\bigcup_{t<T} \mathtt{Train}_t$ , where $\mathtt{Train}_t$ is the full training data for task $t$.
            - Train model `M2` on $\bigcup_{t\leq T} \mathtt{Train}_t$ , where $\mathtt{Train}_t$ is the full training data for task $t$.
        - if the `use_memory_sets` flag is set to `True`, then we construct memory sets and evaluate all models:
            - Models weights for `M1`, `M2` are loaded 
            - Using `memory_set_manager`, create memory sets $\mathcal{M} = \{\mathtt{memory}_t\}_{t<T}\,$ for all tasks save for the terminal one. If gradients are required, gradients from `M1` is used.
            - Train model `M3` on $\bigcup_{t< T}\mathtt{memory}_t \,\cup\, \mathtt{Train}_T$.
            - Evaluate `M2` and `M3` on $\bigcup_{t<T} \mathtt{Test}_t$ , where $\mathtt{Test}_t$ is the full test data for task $t$.
            - Compute the gradient of the loss for `M1`, with respect to the weights $w_{M1}$, evaluated at the union of the memory sets and the optimized weights $w^*_{M1}$ of the model :
$$\mathtt{grad}^{\mathtt{mem}}_{M1} = \nabla_{w_{M1}} \ell\left( w_{M1};\;\bigcup_{t< T}\mathtt{memory}_t \right)\bigg|_{w^*_{M1}}$$
            - Compute the gradient of the loss for `M2`, with respect to the weights $w_{M2}$, evaluated at the union of the memory sets:
$$\mathtt{grad}^{\mathtt{mem}}_{M2} = \nabla_{w_{M2}} \ell\left( w_{M2};\;\bigcup_{t< T}\mathtt{memory}_t \right)\bigg|_{w^*_{M2}}$$
            - Compute the gradient of the loss for `M1`, with respect to the weights $w_{M1}$, evaluated at the union of the training sets and the optimized weights $w^*_{M1}$ of the model :
$$\mathtt{grad}^{\mathtt{full}}_{M1} = \nabla_{w_{M1}} \ell\left( w_{M1};\;\bigcup_{t< T}\mathtt{Train}_t \right)\bigg|_{w^*_{M1}}$$
            - Compute the gradient of the loss for `M2`, with respect to the weights $w_{M2}$, evaluated at the union of the training sets:
$$\mathtt{grad}^{\mathtt{full}}_{M2} = \nabla_{w_{M2}} \ell\left( w_{M2};\;\bigcup_{t< T}\mathtt{Train}_t \right)\bigg|_{w^*_{M2}}$$
            - Compute $\|\mathtt{grad}^{\mathtt{full}}_{M2} -  \mathtt{grad}^{\mathtt{full}}_{M1}\|_2$, and $\|\mathtt{grad}^{\mathtt{mem}}_{M2} - \mathtt{grad}^{\mathtt{mem}}_{M1}\|_2$

**Methods for Creating Task Data**
1. `[Dataset Name].py` contains the pipeline for downloading the data, creating tasks, as well as instantiating models and memory set managers.
    - `[Dataset Name].py` takes in a set of experimental parameters: 
        {T, random seed, p, memory selection method parameters, model training parameters}
    - If `train_full_only` is set to `True`:  `train_task.py` is called to train `M1` and `M2` only. If `model_PATH` is provided, then model weights are saved.
    - If `train_full` is set to `False`: `train_task.py` is called to create memory sets, 
and evaluate all models. `[Dataset Name].py` returns evaluation metrics as a dictionary. If  `model_PATH` is provided, then weights for `M1` and `M2` are loaded from memory. 
    - `[Dataset Name].py` runs the following memory selection methods in serial:
         - `RandomMemorySetManager`
         - `KMeansMemorySetManager`
         - `LambdaMemorySetManager`
         - `GSSMemorySetManager`
         - `iCaRL` with iCaRl loss
         - `iCaRL` with replay loss
2. `run.py` contains the script to automatically generate experimental parameters and run the experimental pipeline
    - Modify only the variable `QUEUE` and the content of `run()`.

### Running Experiments

To run experiments, do the following steps:
1. Define experiment parameters we want to grid search over by modifying `QUEUE` in `run.py`
2. Log into Odyssey
3. Run `python submit_batch.py` in your terminal. 
    - First set up an interactive job first to test out the script. You'll need to set `DRYRUN = True` in `run.py`
    - After debugging issues, set `DRYRUN = False` and submit using `python submit_batch.py`
4. You can load results saved from your runs into a `pandas` dataframe by running `load_results.py` 
