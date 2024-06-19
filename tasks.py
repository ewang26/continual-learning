from jaxtyping import Float
from torch import Tensor
from typing import Set
from data import MemorySetManager
import numpy as np
from torch import Tensor
import torch
from torch import nn


class Task:
    """Class storing all the data for a certain task in a continual learning setting.

    Every task contains some gold standard train_set, all the information you would want for that task,
    then some test_set, which is used to evaluate the model on that task, and a memory set, which is used
    when the task is not the current primary continual learning task, but instead is in the past.
    """

    def __init__(
        self,
        train_x: Float[Tensor, "n f"],
        train_y: Float[Tensor, "n 1"],
        test_x: Float[Tensor, "m f"],
        test_y: Float[Tensor, "m 1"],
        task_labels: Set[int],
        memory_set_manager: MemorySetManager,
    ) -> None:
        """
        Args:
            train_x: The training examples for this task.
            train_y: The training labels for this task.
            test_x: The test examples for this task.
            test_y: The test labels for this task.
            task_labels: Set of labels that this task uses.
            memory_set_manager: The memory set manager to use to create the memory set.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.task_labels = task_labels

        self.memory_x, self.memory_y = memory_set_manager.create_memory_set(
            train_x, train_y
        )
        self.task_labels = task_labels
        self.active = False

        print("IN TASKS.PY")
        print(f"Train memory set shape in tasks.py is: {self.memory_x.shape}")
        print(f"Train memory set length in tasks.py is: {len(self.memory_y)}")

        self.train_weights = torch.ones(self.train_x.shape[0])
        self.memory_set_weights = torch.ones(self.memory_x.shape[0])
        self.test_weights = torch.ones(self.test_x.shape[0])

        if memory_set_manager.__class__.__name__ == 'GSSMemorySetManager':
            self.memory_set_manager = memory_set_manager # save the manager for future use
            self.C_arr = np.array([]) # initialize score array for memory set. i can use to initiaze for weights. 
        
        if memory_set_manager.__class__.__name__ == 'LambdaMemorySetManager':
            self.memory_set_manager = memory_set_manager

        if memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            self.memory_set_manager = memory_set_manager
            # self.train_weights = torch.ones(self.train_x.shape[0])
            # self.memory_set_weights = torch.ones(self.memory_x.shape[0])
            print("empty memory set weights initialized in tasks.py")
            print("Memory set weights shape in tasks: ", self.memory_set_weights)
            # self.memory_z = torch.empty(0)

        print("Memory created in tasks.py with number of samples (?): ", len(self.memory_x))

    def modify_memory(self, sample_x, sample_y, outputs=None, grad_sample=None, grad_batch=None):

        if self.memory_set_manager.__class__.__name__ == 'LambdaMemorySetManager':
            self.memory_x, self.memory_y = self.memory_set_manager.update_memory_lambda(
                self.memory_x, self.memory_y, sample_x, sample_y, outputs
            )

        elif self.memory_set_manager.__class__.__name__ == 'GSSMemorySetManager':
            self.memory_x, self.memory_y, self.C_arr = self.memory_set_manager.update_GSS_greedy(
                self.memory_x, 
                self.memory_y, 
                self.C_arr, 
                sample_x,
                sample_y,
                grad_sample, 
                grad_batch) #update buffer + scores
            
        else:
            raise NotImplementedError("Only Lambda and GSS Memory Selection methods update memory set in runtime.")
        

    #GCR functions
    def update_memory_set_weights(self, weights):
        if self.memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            # print("weights that have been passed in tasks:")
            # print(weights)
            self.memory_set_weights = weights
        
            #print("Memory set weights updated in tasks.py")
            #print("Memory set weights shape in tasks: ", self.memory_set_weights)
        else:
            raise NotImplementedError("Only GCR Memory Selection method updates memory set weights in runtime.")
    
    def update_task_memory_x(self, x):
        if self.memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            self.memory_x = x
        else:
            raise NotImplementedError("Only GCR Memory Selection method updates memory set x in tasks.py")

    def update_task_memory_y(self, y):
        if self.memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            self.memory_y = y
        else:
            raise NotImplementedError("Only GCR Memory Selection method updates memory set y in tasks.py")

        
    def get_memory_set_weights(self):
        if self.memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            return self.memory_set_weights
        else:
            raise NotImplementedError("Only GCR Memory Selection method returns memory set weights in runtime.")
        
    def get_train_weights(self):
        if self.memory_set_manager.__class__.__name__ == 'GCRMemorySetManager':
            return self.train_weights
        else:
            raise NotImplementedError("Only GCR Memory Selection method returns memory set weights in runtime.")

