from typing import Dict, Union

from data import RandomMemorySetManager, KMeansMemorySetManager, LambdaMemorySetManager, \
    GSSMemorySetManager, ClassBalancedReservoirSampling, iCaRL, GCRMemorySetManager
from managers import (
    MnistManagerSplit,
    Cifar10ManagerSplit,
    Cifar100ManagerSplit,
    Cifar10Full,
)
from pathlib import Path
from dataclasses import dataclass
from models import MLP, CifarNet
import torch
import random
import numpy as np


@dataclass
class Config:
    def __init__(self, config_dict: Dict[str, Union[str, int, float]]):
        self.config_dict = config_dict
        
        # for batch training and evaluation
        self.p_arr = np.array(self.config_dict['p_arr'])
        self.num_samples = np.array(self.config_dict['num_samples'])
        self.memory_selection_method = self.config_dict['memory_set_manager']
        self.use_random_img = self.config_dict['use_random_img']
        self.num_ideal_models = self.config_dict['num_ideal_models']
        #print(self.p_arr)
        
        #k-means addtions
        self.num_centroids = config_dict.get('num_centroids', 10)  # Default value is 10
        self.num_classes = config_dict.get('num_classes', 10)     # Default value is 10
        self.device = torch.device(config_dict.get('device', 'cpu'))  # Default value is 'cpu'        

        # debugging config
        self.train_debug = self.config_dict['train_debug']

        # String run_name for wandb / logfiles
        self.run_name = (
            f"Manager.{config_dict['learning_manager']}_"
            f"MemorySetManager.{config_dict['memory_set_manager']}_p.{config_dict['p']}_"
            f"Model.{config_dict['model']['type']}"
        )

        if "random_seed" in config_dict:
            print("Seed given in config, setting deterministic run")
            random_seed = config_dict["random_seed"]
            self.random_seed = random_seed

            # Set run to be deterministic
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            #torch.backends.cudnn.benchmark = False
            #torch.use_deterministic_algorithms(True)

        # Pass config into python
        for key, val in config_dict.items():
            if key == "memory_set_manager":
                if val == "random":
                    setattr(self, key, RandomMemorySetManager)
                elif val == "kmeans":
                    if config_dict['learning_manager'] == 'mnist_split':
                        setattr(self, key, KMeansMemorySetManager)
                    elif config_dict['learning_manager'] == 'cifar10_split':
                        setattr(self, key, KMeansMemorySetManager)
                    else:
                        print('is this method implemented fsor kmeans?')
                        assert False
                elif val == "lambda" or val == "Lambda":
                    setattr(self, key, LambdaMemorySetManager)
                elif val == "GSS":
                    setattr(self, key, GSSMemorySetManager)
                elif val == "class_balanced":
                    setattr(self, key, ClassBalancedReservoirSampling)
                elif val == "iCaRL":
                    setattr(self, key, iCaRL)
                elif val == "GCR":
                    setattr(self, key, GCRMemorySetManager)
                else:
                    raise ValueError(
                        f"{val} memory set manager is not valid"
                    )
            elif key == "learning_manager":
                if val == "mnist_split":
                    setattr(self, key, MnistManagerSplit)
                elif val == "cifar10_split":
                    setattr(self, key, Cifar10ManagerSplit)
                elif val == "cifar100_split":
                    setattr(self, key, Cifar100ManagerSplit)
                elif val == "cifar10_full":
                    setattr(self, key, Cifar10Full)
                else:
                    raise ValueError(
                        f"{val} learning manager is not valid"
                    )
            elif key == "model":
                model_type = val["type"]
                model_params = val["params"]
                if model_type == "mlp":
                    setattr(self, key, MLP(**model_params))
                elif model_type == "cnn":
                    setattr(self, key, CifarNet(**model_params))
                    self.model.weight_init()
                else:
                    raise ValueError(
                        f"{model_type} model is not valid"
                    )

                # Set model name 
                self.model_name = model_type
                for val in model_params.values():
                    self.model_name += f"_{val}"
            elif key == "model_load_dir":
                self.model_load_dir = Path(val)
            elif key == "model_save_dir":
                self.model_save_dir = Path(val)
            elif key == "experiment_metadata_path": 
                self.experiment_metadata_path = Path(val)
            elif key == "grad_save_dir":
                self.grad_save_dir = Path(val)
            else:
                setattr(self, key, val)

        if getattr(self, "model_load_dir", None) is not None:
            self.run_name = f"EVAL_" + self.run_name
        else:
            self.run_name = f"TRAIN_" + self.run_name

def dict_to_config_string(config_dict: Dict[str, Union[str, int, float]]) -> str:
    # Convert config dict to string for wandb
    config_string = ""
    for key, val in config_dict.items():
        config_string += f"{key}.{val}_"

    if config_string[-1] == "_":
        config_string = config_string[:-1]
    return config_string