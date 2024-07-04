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
        train_x,
        train_y,
        test_x,
        test_y,
        task_labels,
        memory_set_manager,
        random_seed = 1,
        class_balanced = True
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
        # Set random seeds
        self.rand = RandomState(random_seed)
        self.generator = torch.Generator().manual_seed(random_seed)
        torch.manual_seed(random_seed)
        # Data for the task
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.task_labels = task_labels
        # Memory sets and memory set manager
        self.memory_set_manager = memory_set_manager
        self.class_balanced = class_balanced
        self.memory_x, self.memory_y = self.memory_set_manager.create_memory_set(
            train_x, train_y
        )
        # Weights
        self.memory_set_weights = torch.ones(self.memory_x.shape[0])
        self.train_weights = torch.ones(self.train_x.shape[0])
        self.test_weights = torch.ones(self.test_x.shape[0])
        # Additional attributes required for special memory set selection methods
        if memory_set_manager.__class__.__name__ == 'GSSMemorySetManager':
            self.similarity_scores = np.empty(0) # initialize score array for memory set 

    # Compute gradient per sample
    def __compute_grad__(self, classifier, criterion, sample, target):
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)
        outputs = classifier(sample)
        loss = criterion(outputs, target)
        structured_grad = torch.autograd.grad(loss, list(classifier.parameters()))
        flatten_grad = [layer.flatten() for layer in structured_grad]
        flatten_grad = torch.cat(flatten_grad, 0).reshape((1, -1))
        return flatten_grad

    # Get per sample gradients for a batch
    def __compute_sample_grads__(self, classifier, data, targets):
        batch_size = data.shape[0]
        sample_grads = [compute_grad(classifier, data[i, :].reshape(reshape), targets[i]) for i in range(batch_size)]
        sample_grads = torch.cat(sample_grads, 0)
        return sample_grads

    def update_memory_sets(self, model, criterion, debug_mode=False):
        # Update GSS memory set
        if memory_set_manager.__class__.__name__ == 'GSSMemorySetManager':
            # Keep track of number of replacements for debugging
            if debug_mode:
                replacement_counter = 0 

            shuffled_idx = rand.permutation(self.train_x.shape[0])
            for i in range(train_x.shape[0]):
                idx = shuffled_idx[i]
                grad_sample = self.__compute_grad__(model, criterion, train_x[idx], train_y[idx])
                if memory_x.shape[0] == 0:
                    grad_batch = self.__compute_sample_grads__(model, train_x[:2], train_y[:2])
                else:
                    grad_batch = self.__compute_sample_grads__(model, self.memory_x, self.memory_y)

                prv_scores = self.similarity_scores.copy()

                self.memory_x, self.memory_y, self.similarity_scores = mset_manager.update_GSS_greedy(self.memory_x, self.memory_y, 
                                                                                                      self.similarity_scores, 
                                                                                                      train_x[idx].reshape((1, -1)), train_y[idx].reshape((-1,)), 
                                                                                                      grad_sample, grad_batch, class_balanced=self.class_balanced)
                # Keep track of number of replacements for debugging
                if debug_mode:
                    if i % 100 == 0:
                        print(f'GSS pass: {i}')

                    if i > mset_manager.memory_set_size:
                        if len(self.similarity_scores) != len(prv_scores):
                            replacement_counter += 1
                        else:
                            diff = np.linalg.norm(self.similarity_scores - prv_scores)
                            if diff > 0:
                                replacement_counter += 1
            # Keep track of number of replacements for debugging
            if debug_mode:
                print(f'Number of GSS replacements: {replacement_counter}')

        # Update Lambda memory set
        if memory_set_manager.__class__.__name__ == 'LambdaMemorySetManager':
            self.memory_x, self.memory_y = mset_manager.update_memory(self.memory_x, self.memory_y, self.train_x, self.train_y, output, class_balanced=True)

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

