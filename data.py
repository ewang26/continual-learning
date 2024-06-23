from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
import pdb
# pdb.set_trace()


import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms.functional import to_tensor

from torchvision.models import resnet18
import torchvision.transforms as transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Check for M1 Mac MPS (Apple Silicon GPU) support
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Using M1 Mac")
    DEVICE = torch.device("mps")
# Check for CUDA support (NVIDIA GPU)
elif torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
# Default to CPU if neither is available
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")

# DEVICE = torch.device("mps") # change this before cluster!
torch.autograd.set_detect_anomaly(True)


class MemorySetManager(ABC):
    @abstractmethod
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]):
        """Creates a memory set from the given dataset. The memory set is a subset of the dataset."""
        pass


class RandomMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)

    # Randomly select elements from the dataset to be in the memory set.
    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Creates random memory set.

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """

        memory_set_size = int(x.shape[0] * self.p)
        # Select memeory set random elements from x and y, without replacement
        memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[
            :memory_set_size
        ]
        #print(memory_set_indices)
        memory_x = x[memory_set_indices]
        memory_y = y[memory_set_indices]

        # print(f"Shape of memory set in data.py is: {memory_x.shape}") # erik was just debugging some stuff
        # print(f"Memory set size in data.py is: {memory_set_size}")
        # print("random memory set created in data.py")
        return memory_x, memory_y

#THEODORA K-MEANS
class KMeansMemorySetManager(MemorySetManager):
    def __init__(self, p: float, num_centroids: int, device: torch.device, random_seed: int = 42):
        """
        Args:
            p: The percentage of samples to retain in the memory set.
            num_centroids: The number of centroids to use for K-Means clustering.
            device: The device to use for computations (e.g., torch.device("cuda")).
            random_seed: The random seed for reproducibility.
        """
        self.p = p
        self.num_centroids = num_centroids
        self.device = device
        self.random_seed = random_seed
        
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Initialize dictionaries to store centroids, cluster counters, and memory sets for each class
        self.centroids = {}
        self.cluster_counters = {}
        self.memory_sets = {}
        
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m 1"]]:
        """Creates memory set using K-Means clustering for each class.
        
        Args:
            x: x data.
            y: y data.
        
        Returns:
            (memory_x, memory_y) tuple, where memory_x and memory_y are tensors.
        """
        device = x.device
        n = x.shape[0]
        f = x.shape[1]
        memory_size = int(n * self.p)
        
        # Get unique classes
        classes = torch.unique(y).tolist()
        num_classes = len(classes)
        
        # Calculate the memory size per class
        memory_size_per_class = memory_size // num_classes
        
        # Initialize memory arrays for each class
        memory_x = {}
        memory_y = {}
        memory_distances = {}
        self.memory_set_indices = {}
        
        for class_label in classes:
            memory_x[class_label] = torch.zeros(memory_size_per_class, f, device=device)
            memory_y[class_label] = torch.zeros(memory_size_per_class, 1, dtype=torch.long, device=device)
            memory_distances[class_label] = torch.full((memory_size_per_class,), float("inf"), device=device)
            self.memory_set_indices[class_label] = torch.zeros(memory_size_per_class, dtype=torch.long, device=device)
        
        # Iterate over each class
        for class_label in classes:
            # Get samples and labels for the current class
            class_mask = (y == class_label).squeeze()
            class_samples = x[class_mask]
            class_labels = y[class_mask]
            
            # Initialize centroids and cluster counters for the current class if not already initialized
            if class_label not in self.centroids:
                self.centroids[class_label] = torch.randn(self.num_centroids, f, device=device)
                self.cluster_counters[class_label] = torch.zeros(self.num_centroids, device=device)
            
            # Iterate over the samples of the current class
            for i in range(class_samples.shape[0]):
                sample = class_samples[i]
                label = class_labels[i].item()
                
                # Find the closest centroid for the current class
                distances = torch.sqrt(torch.sum((self.centroids[class_label] - sample) ** 2, dim=1))
                closest_centroid_idx = torch.argmin(distances).item()
                
                # Update the cluster counter and centroid for the current class
                self.cluster_counters[class_label][closest_centroid_idx] += 1
                self.centroids[class_label][closest_centroid_idx] += (sample - self.centroids[class_label][closest_centroid_idx]) / self.cluster_counters[class_label][closest_centroid_idx]
                
                # Update the memory set for the current class
                class_memory_size = memory_x[class_label].shape[0]
                distance = distances[closest_centroid_idx]
                if i < class_memory_size:
                    memory_x[class_label][i] = sample
                    memory_y[class_label][i] = label
                    memory_distances[class_label][i] = distance
                    self.memory_set_indices[class_label][i] = i
                else:
                    max_idx = torch.argmax(memory_distances[class_label])
                    if distance < memory_distances[class_label][max_idx]:
                        memory_x[class_label][max_idx] = sample
                        memory_y[class_label][max_idx] = label
                        memory_distances[class_label][max_idx] = distance
                        self.memory_set_indices[class_label][max_idx] = i
        
        # Concatenate memory sets from all classes
        memory_x_concat = torch.cat(list(memory_x.values()), dim=0)
        memory_y_concat = torch.cat(list(memory_y.values()), dim=0).view(-1)
        
        return memory_x_concat, memory_y_concat
    

# #this is for cifar rn 
# class KMeansMemorySetManager(MemorySetManager):
#     def __init__(self, p: float, num_centroids: int, device: torch.device, random_seed: int = 42):
#         self.p = p
#         self.num_centroids = num_centroids
#         self.random_seed = random_seed
#         torch.manual_seed(self.random_seed)
#         self.centroids = {}
#         self.cluster_counters = {}
#         self.memory_sets = {}
        
#     def create_memory_set(self, x: Float[Tensor, "n c h w"], y: Float[Tensor, "n 1"]) -> Tuple[Float[Tensor, "m c h w"], Float[Tensor, "m 1"]]:
#         if self.p == 1:  # Check if p is set to 1
#             return x, y  # Return the entire dataset as the memory set
        
#         n, c, h, w = x.shape
#         memory_size = int(n * self.p)
        
#         # Get unique classes
#         classes = torch.unique(y).tolist()
#         num_classes = len(classes)
        
#         # Calculate the memory size per class
#         memory_size_per_class = memory_size // num_classes
        
#         # Initialize memory arrays for each class
#         memory_x = {}
#         memory_y = {}
#         memory_distances = {}
#         self.memory_set_indices = {}
        
#         for class_label in classes:
#             memory_x[class_label] = torch.zeros((memory_size_per_class, c, h, w))
#             memory_y[class_label] = torch.zeros((memory_size_per_class, 1), dtype=torch.long)
#             memory_distances[class_label] = torch.full((memory_size_per_class,), float("inf"))
#             self.memory_set_indices[class_label] = torch.zeros(memory_size_per_class, dtype=torch.long)
        
#         # Iterate over each class
#         for class_label in classes:
#             class_mask = (y == class_label).squeeze()
#             class_samples = x[class_mask]
#             class_labels = y[class_mask]
            
#             if class_label not in self.centroids:
#                 self.centroids[class_label] = torch.randn((self.num_centroids, c, h, w))
#                 self.cluster_counters[class_label] = torch.zeros(self.num_centroids)
            
#             # Iterate over the samples of the current class
#             for i in range(class_samples.shape[0]):
#                 sample = class_samples[i].unsqueeze(0)  # Add batch dimension
#                 # Compute distances using broadcasting, sum over spatial and channel dimensions
#                 distances = torch.sqrt(torch.sum((self.centroids[class_label] - sample) ** 2, dim=[1, 2, 3]))
#                 closest_centroid_idx = torch.argmin(distances).item()
                
#                 self.cluster_counters[class_label][closest_centroid_idx] += 1
#                 # Update centroids with learning rate based on cluster size
#                 learning_rate = 1 / self.cluster_counters[class_label][closest_centroid_idx]
#                 self.centroids[class_label][closest_centroid_idx] *= (1 - learning_rate)
#                 self.centroids[class_label][closest_centroid_idx] += learning_rate * sample.squeeze(0)
                
#                 distance = distances[closest_centroid_idx]
#                 if i < memory_size_per_class:
#                     memory_x[class_label][i] = sample.squeeze(0)
#                     memory_y[class_label][i] = class_labels[i]
#                     memory_distances[class_label][i] = distance
#                     self.memory_set_indices[class_label][i] = i
#                 else:
#                     max_idx = torch.argmax(memory_distances[class_label])
#                     if distance < memory_distances[class_label][max_idx]:
#                         memory_x[class_label][max_idx] = sample.squeeze(0)
#                         memory_y[class_label][max_idx] = class_labels[i]
#                         memory_distances[class_label][max_idx] = distance
#                         self.memory_set_indices[class_label][max_idx] = i
        
#         # Concatenate memory sets from all classes
#         memory_x_concat = torch.cat(list(memory_x.values()), dim=0)
#         memory_y_concat = torch.cat(list(memory_y.values()), dim=0).view(-1)
        
#         return memory_x_concat, memory_y_concat


#new kmeans for cuda

# class KMeansMemorySetManager(MemorySetManager):
#     def __init__(self, p: float, num_centroids: int, device: torch.device, random_seed: int = 42):
#         self.p = p
#         self.num_centroids = num_centroids
#         self.device = device  # Store device
#         self.random_seed = random_seed
#         torch.manual_seed(self.random_seed)
#         self.centroids = {}
#         self.cluster_counters = {}
#         self.memory_sets = {}
        
#     def create_memory_set(self, x: Float[Tensor, "n c h w"], y: Float[Tensor, "n 1"]) -> Tuple[Float[Tensor, "m c h w"], Float[Tensor, "m 1"]]:
#         if self.p == 1:  # Check if p is set to 1
#             return x.to(self.device), y.to(self.device)  # Return the entire dataset as the memory set
        
#         n, c, h, w = x.shape
#         memory_size = int(n * self.p)
        
#         # Get unique classes
#         classes = torch.unique(y).tolist()
#         num_classes = len(classes)
        
#         # Calculate the memory size per class
#         memory_size_per_class = memory_size // num_classes
        
#         # Initialize memory arrays for each class
#         memory_x = {}
#         memory_y = {}
#         memory_distances = {}
#         self.memory_set_indices = {}
        
#         for class_label in classes:
#             memory_x[class_label] = torch.zeros((memory_size_per_class, c, h, w), device=self.device)
#             memory_y[class_label] = torch.zeros((memory_size_per_class, 1), dtype=torch.long, device=self.device)
#             memory_distances[class_label] = torch.full((memory_size_per_class,), float("inf"), device=self.device)
#             self.memory_set_indices[class_label] = torch.zeros(memory_size_per_class, dtype=torch.long, device=self.device)
        
#         # Iterate over each class
#         for class_label in classes:
#             class_mask = (y == class_label).squeeze()
#             class_samples = x[class_mask].to(self.device)
#             class_labels = y[class_mask].to(self.device)
            
#             if class_label not in self.centroids:
#                 self.centroids[class_label] = torch.randn((self.num_centroids, c, h, w), device=self.device)
#                 self.cluster_counters[class_label] = torch.zeros(self.num_centroids, device=self.device)
            
#             # Iterate over the samples of the current class
#             for i in range(class_samples.shape[0]):
#                 sample = class_samples[i].unsqueeze(0)  # Add batch dimension
#                 # Compute distances using broadcasting, sum over spatial and channel dimensions
#                 distances = torch.sqrt(torch.sum((self.centroids[class_label] - sample) ** 2, dim=[1, 2, 3]))
#                 closest_centroid_idx = torch.argmin(distances).item()
                
#                 self.cluster_counters[class_label][closest_centroid_idx] += 1
#                 # Update centroids with learning rate based on cluster size
#                 learning_rate = 1 / self.cluster_counters[class_label][closest_centroid_idx]
#                 self.centroids[class_label][closest_centroid_idx] *= (1 - learning_rate)
#                 self.centroids[class_label][closest_centroid_idx] += learning_rate * sample.squeeze(0)
                
#                 distance = distances[closest_centroid_idx]
#                 if i < memory_size_per_class:
#                     memory_x[class_label][i] = sample.squeeze(0)
#                     memory_y[class_label][i] = class_labels[i]
#                     memory_distances[class_label][i] = distance
#                     self.memory_set_indices[class_label][i] = i
#                 else:
#                     max_idx = torch.argmax(memory_distances[class_label])
#                     if distance < memory_distances[class_label][max_idx]:
#                         memory_x[class_label][max_idx] = sample.squeeze(0)
#                         memory_y[class_label][max_idx] = class_labels[i]
#                         memory_distances[class_label][max_idx] = distance
#                         self.memory_set_indices[class_label][max_idx] = i
        
#         # Concatenate memory sets from all classes
#         memory_x_concat = torch.cat(list(memory_x.values()), dim=0)
#         memory_y_concat = torch.cat(list(memory_y.values()), dim=0).view(-1)
        
#         return memory_x_concat, memory_y_concat


# Jonathan Lambda Method
class LambdaMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p

    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]):
        # initializing memory sets as empty for initial task (which uses all the data)
        # self.memory_set_size = int(x.shape[0] * self.p)
        #return torch.empty(0), torch.empty(0)
        return torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE)


    def update_memory_lambda(self, memory_x,  memory_y, sample_x, sample_y, outputs):
        """
        Function to update the memory buffer in Lambda Memory Selection.

        Args:
            memory_x and memory_y: the existing memory datasets.
            sample_x and sample_y: the full data from the terminal task.
            outputs: tensor of size [n x k] where n is number of samples in sample_x or sample_y, and k is number of classes to classify into.
                Outputs of forward pass through the network of all data in sample_x.
        
        Returns:
            memory_x and memory_y.long(): new memory datasets including the memory dataset for the existing task.
        """
        terminal_task_size = outputs.shape[0]
        trace_list = []
        for i in range(terminal_task_size):
            # take output layer and apply softmax to get probabilities of classification for each output
            class_p = torch.softmax(outputs[i], dim=0)

            # create a matrix of p @ (1-p).T to represent decision uncertainty at each class
            #decision_uncertainty = torch.ger(class_p, (1 - class_p).T)
            decision_uncertainty = torch.ger(class_p, (1 - class_p))

            # calculate the trace of this matrix to assess the uncertainty in classification across multiple classes
            # the trace equivalent to the hessian of the loss wrt the output layer
            decision_trace = torch.trace(decision_uncertainty)
            # print(decision_trace)
            trace_list.append(decision_trace.item())
        print(trace_list[:10])
        # calculate size of memory set to create 
        #note: this does class balancing if data in the tasks are already balanced
            # more work must be done to create constant memory size for each class regardless of initial class distribution in task space
        memory_size = int(terminal_task_size*self.p)

        # getting indexes of the highest trace 
        argsorted_indx = sorted(range(len(trace_list)), key=lambda x: trace_list[x], reverse=True)
        desired_indx = argsorted_indx[:memory_size]
        # print(sample_x[desired_indx][:5])
        idx = desired_indx[0]
        # print(sample_x[0])

        # finding the memory set of terminal task and concatenating it to the existing memory set
        memory_x = torch.cat((memory_x, sample_x[desired_indx].to(DEVICE)))
        memory_y = torch.cat((memory_y, sample_y[desired_indx].to(DEVICE)))
        return memory_x, memory_y.long()


# Alan Gradient Sample Selection (GSS)
class GSSMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: fraction of task dataset to be included in replay buffer.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)
        self.gss_p = 0.1
        np.random.seed(random_seed)

    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Initializes an empty memory replay buffer if training, called when task objects are created
        Else, use ideal model to generate GSS memory set

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        #start out memory buffer with p*task_data_length
        self.memory_set_size = int(x.shape[0] * self.p)
        self.memory_set_inc = self.memory_set_size
        if self.p == 1:
            return x, y
        # # Select memeory set random elements from x and y, without replacement
        # memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[
        #     :self.memory_set_size
        # ]
        # #print(memory_set_indices)
        # memory_x = x[memory_set_indices]
        # memory_y = y[memory_set_indices]

        # return memory_x, memory_y
        return torch.empty(0), torch.empty(0)
    
    def update_GSS_greedy(self, memory_x, memory_y, C_arr, sample_x, sample_y, grad_sample, grad_batch):
        '''
        TODO implement alg 2 in paper here
        memory_x,y = current memory set for the task (to be used in later tasks)
        C_arr = current list of corresponding scores of each element in memory set
        sample_x,y = new sample
        grad_sample = gradient of new sample
        grad_batch = gradent of random batch of memory_x,y
        '''
        if self.p == 1:
            return memory_x, memory_y.long(), C_arr
        # first case, if we dont reach maximum size of memory set, just add it
        # if memory_x.shape[0] + 1 >= self.memory_set_size:
        #     print('in gss greedy')
        #     print(memory_x.shape)
        #     print(memory_y.shape)
        #     print(C_arr.shape)

        sample_norm, batch_norm = np.linalg.norm(grad_sample), np.linalg.norm(grad_batch)
        if self.memory_set_size == 0:
            c = 0
        else:
            c = ((np.dot(grad_sample, grad_batch) / (sample_norm*batch_norm)) + 1) if not (sample_norm*batch_norm == 0) else 1 # else dont add it
        if (memory_x.shape[0] < self.memory_set_size) and (0 <= c):
            memory_x = torch.cat((memory_x, sample_x), 0)
            memory_y = torch.cat((memory_y, sample_y), 0)
            C_arr = np.concatenate((C_arr, np.array([c])), 0)
        else:
            if 0 <= c < 1:
                P = C_arr / np.sum(C_arr)
                i = np.random.choice(np.arange(self.memory_set_size), p = P)
                r = np.random.rand()
                if r < C_arr[i] / (C_arr[i] + c):
                    memory_x[i] = sample_x
                    memory_y[i] = sample_y
                    C_arr[i] = c
                #     print('replaced!')
                # else:
                #     print('no replace!')

        # if memory_x.shape[0] + 1 >= self.memory_set_size:
        #     print('after update')
        #     print(memory_x.shape)
        #     print(memory_y.shape)
        #     print(C_arr)
        #     input()
        
        return memory_x, memory_y.long(), C_arr

    def create_memory_set(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the memory set using class-balanced reservoir sampling.
        Args:
            x: Input features as a tensor.
            y: Corresponding labels as a tensor.
        Returns:
            A tuple containing tensors for the memory set's features and labels.
        """
        self.memory_set_size = int(x.shape[0] * self.p)
        # reset memory and counts
        self.memory_x = x.new_empty((0, *x.shape[1:]))
        self.memory_y = y.new_empty((0,), dtype=torch.long)
        self.class_counts_in_memory = {}
        self.stream_class_counts = {}
        self.full_classes = set()

        for i in range(x.shape[0]):
            self.update_memory_set(x[i], y[i])

        return self.memory_x, self.memory_y        

class ClassBalancedReservoirSampling:
    def __init__(self, p: float, random_seed: int = 42):
        """
        Initializes the sampling process.
        Args:
            p: Probability of an element being in the memory set.
            random_seed: Seed for the random number generator to ensure reproducibility.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)
        self.memory_x = torch.Tensor().new_empty((0,)) 
        self.memory_y = torch.Tensor().new_empty((0,), dtype=torch.long)
        self.class_counts_in_memory = {}
        self.stream_class_counts = {}
        self.memory_set_size = 0
        self.full_classes = set()

    def update_memory_set(self, x_i: torch.Tensor, y_i: torch.Tensor):
        """
        Updates the memory set with the new instance (x_i, y_i), following the reservoir sampling algorithm.
        Args:
            x_i: The instance of x data.
            y_i: The instance of y data (class label).
        """
        y_i_item = y_i.item()

        self.stream_class_counts[y_i_item] = self.stream_class_counts.get(y_i_item, 0) + 1

        if self.memory_y.numel() < self.memory_set_size:
            # memory is not filled, so we add the new instance
            self.memory_x = torch.cat([self.memory_x, x_i.unsqueeze(0)], dim=0)
            self.memory_y = torch.cat([self.memory_y, y_i.unsqueeze(0)], dim=0)

            # the line below ensures that if y_i_item is not already a key in the dictionary, the method will return 0
            self.class_counts_in_memory[y_i_item] = self.class_counts_in_memory.get(y_i_item, 0) + 1
            # this checks if the class has become full because of the addition
            if len(self.memory_y) == self.memory_set_size:
                largest_class = max(self.class_counts_in_memory, key=self.class_counts_in_memory.get)
                self.full_classes.add(largest_class)
        else:
            # first determine if the class is full. if not, then select and replace an instance from the largest class
            if y_i_item not in self.full_classes:
                # identify the largest class that is considered full
                largest_class_item = max(self.class_counts_in_memory.items(), key=lambda item: item[1])[0]
                indices_of_largest_class = (self.memory_y == largest_class_item).nonzero(as_tuple=True)[0]
                replace_index = indices_of_largest_class[torch.randint(0, len(indices_of_largest_class), (1,), generator=self.generator)].item()
                self.memory_x[replace_index] = x_i
                self.memory_y[replace_index] = y_i

                # update the class counts accordingly
                self.class_counts_in_memory[largest_class_item] -= 1
                self.class_counts_in_memory[y_i_item] = self.class_counts_in_memory.get(y_i_item, 0) + 1

                # check and update full status for replaced class
                if self.class_counts_in_memory[largest_class_item] <= max(self.class_counts_in_memory.values()):
                    self.full_classes.add(max(self.class_counts_in_memory, key=self.class_counts_in_memory.get))
            else:
                # if the class is already full, apply the sampling decision based on mc/nc
                mc = self.class_counts_in_memory[y_i_item]
                nc = self.stream_class_counts[y_i_item]
                if torch.rand(1, generator=self.generator).item() <= mc / nc:
                    indices_of_y_i_class = (self.memory_y == y_i_item).nonzero(as_tuple=True)[0]
                    replace_index = indices_of_y_i_class[torch.randint(0, len(indices_of_y_i_class), (1,), generator=self.generator)].item()
                    self.memory_x[replace_index] = x_i
                    self.memory_y[replace_index] = y_i

    def create_memory_set(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the memory set using class-balanced reservoir sampling.
        Args:
            x: Input features as a tensor.
            y: Corresponding labels as a tensor.
        Returns:
            A tuple containing tensors for the memory set's features and labels.
        """
        self.memory_set_size = int(x.shape[0] * self.p)
        # reset memory and counts
        self.memory_x = x.new_empty((0, *x.shape[1:]))
        self.memory_y = y.new_empty((0,), dtype=torch.long)
        self.class_counts_in_memory = {}
        self.stream_class_counts = {}
        self.full_classes = set()

        for i in range(x.shape[0]):
            self.update_memory_set(x[i], y[i])

        return self.memory_x, self.memory_y        


# Hyper Parameters
num_epochs = 2
batch_size = 64
learning_rate = 0.002

class iCaRLNet(nn.Module):
    def __init__(self, feature_size, n_classes):
        # Network architecture
        super(iCaRLNet, self).__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes)
        self.grayscale_to_rgb = transforms.Compose([transforms.Lambda(lambda x: torch.cat([x, x, x], dim=1))])

        self.n_classes = 0
        self.n_known = 0

        # list containing exemplar_sets
        # each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []
        self.exemplar_labels = []
        self.total_data = []

        # Learning method
        self.cls_loss = nn.BCELoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        #self.optimizer = optim.SGD(self.parameters(), lr=2.0,
        #                           weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    # pass in weights: def forward(self, x, weight=None):
    def forward(self, x):
        # print(f"during forward, x shape is: {x.shape}")
        # breakpoint()
        #WP: let's get the forward function to be able to take in a set of weights
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        #WP: don't forget the biases: self.fc.bias.data
        weight = self.fc.weight.data
        bias = self.fc.bias.data

        #WP: check that adding the new heads is correct (i.e. why bias=False)
        # self.fc = nn.Linear(in_features, out_features+n)
        # self.fc.bias.data[:out_features] = biases
        self.fc = nn.Linear(in_features, out_features+n)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias

        self.n_classes += n


    def construct_exemplar_set(self, images, labels, m, transform):
        """Construct an exemplar set for image set

        Args:
            images: torch tensor containing images of a class
        """

        exemplar_indices = []
        exemplar_set = []
        exemplar_features = []

        features = []
        self.feature_extractor.eval()
        with torch.no_grad():  
            images = images.to(DEVICE)
            features = self.feature_extractor(images)  # Shape [n_images, feature_dim]
            # breakpoint()
            norms = torch.norm(features, dim=1, keepdim=True)  # Shape [n_images, 1]
            normalized_features = features / norms  # broadcasting the division. will have shape [12665, 2048]
        
        features = torch.tensor(normalized_features).to(DEVICE) 
        class_mean = torch.mean(features, axis=0)  
        class_mean_norm = torch.linalg.norm(class_mean) # Do I need to compute the norm of this? class_mean will be a scalar
        class_mean = class_mean / class_mean_norm # normalize

        sum_inner = torch.zeros_like(class_mean).to(DEVICE) # base case for when there are no exemplars yet (k=0)

        for k in range(m):
            min_val = float('inf')
            arg_min_i = None

            for i, feature_x in enumerate(features): # this loop simulates the argmin process
                if i in exemplar_indices:
                    continue
  
                coefficient = 1/(k+1)
                sum_outer = feature_x + sum_inner
                coeff_sum_outer = coefficient * sum_outer

                normed_val = torch.linalg.norm(class_mean - coeff_sum_outer)

                if normed_val < min_val:
                    min_val = normed_val
                    arg_min_i = i
            
            print(f"\narg_min_i is: {arg_min_i}")

            if argmin_i is None:
                print("No suitable element found during construction")
            # breakpoint()
            exemplar_set.append(images[arg_min_i])
            exemplar_features.append(features[arg_min_i])
            exemplar_indices.append(arg_min_i)

            print(f"Shape of exemplar_features is: {torch.stack(exemplar_features).shape}")
            # sum_inner can be updated by adding the new exemplar image (i.e., features[arg_min_i])
            sum_inner = sum_inner + features[arg_min_i]
            # breakpoint()

        # Updating the class variables
        self.exemplar_sets.append(torch.stack(exemplar_set)) # self.exemplar_sets should be a list of tensors
        self.exemplar_labels.append(labels[:m])
        return torch.tensor(exemplar_indices) # only need to return the indices of selected elements from the class data
        # return torch.stack(exemplar_set), labels[:m]


    def combine_exemplar_sets(self):
        self.total_data = [[x, y] for x, y in zip(self.exemplar_sets, self.exemplar_labels)]


    def update_representation(self, x, y, p, loss_type):
        """Update the network representation using a tensor of images (x) and their corresponding labels (y)."""
        self.compute_means = True
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Identify and increment classes
        unique_classes = set(y.cpu().numpy().tolist())
        new_classes = [cls for cls in unique_classes if cls >= self.n_classes]

        self.increment_classes(len(new_classes))
        self.to(DEVICE)
        print(f"{len(new_classes)} new classes.")

        self.combine_exemplar_sets()
        
        if self.total_data: # if the exemplar sets are not empty
            print("\nnot empty")
            exemplar_xs = torch.cat(self.exemplar_sets, dim=0)
            exemplar_ys = torch.cat(self.exemplar_labels, dim=0).to(DEVICE)

        else: # if the exemplar sets are not empty
            # initialize exemplar_xs and exemplar_ys as empty tensors with the remaining dimensions according to x and y
            exemplar_xs = torch.empty((0, *x.shape[1:]), dtype=x.dtype, device=x.device)
            exemplar_ys = torch.empty((0, *y.shape[1:]), dtype=y.dtype, device=y.device)
        
        # concatenate memory and task data
        all_xs = torch.cat((exemplar_xs, x), dim=0)
        all_ys = torch.cat((exemplar_ys, y), dim=0)

        print(f"exemplar_xs shape is {exemplar_xs.shape}, x shape is {x.shape}")
        print(f"exemplar_ys shape is {exemplar_ys.shape}, y shape during update is {y.shape}")

        print(f"all_xs shape: {all_xs.shape}")
        combined_dataset = torch.utils.data.TensorDataset(all_xs, all_ys)
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        print(f"Shapes are: {all_xs.shape}, {all_ys.shape}")
        print(exemplar_xs.shape)

        all_dataset = TensorDataset(all_xs, all_ys)
        all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True) # batch_size = 64

        exemplar_dataloader = None
        if exemplar_xs.size(0) > 0: 
            exemplar_dataset = TensorDataset(exemplar_xs, exemplar_ys)
            exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=batch_size, shuffle=True)

        optimizer = self.optimizer

        # network training
        if loss_type == "replay": # replay loss
            if exemplar_dataloader:
                for epoch in range(num_epochs):
                    print(f"epoch is {epoch}")
                    for (batch_ex_xs, batch_ex_ys), (batch_xs, batch_ys) in zip(exemplar_dataloader, all_dataloader):
                        optimizer.zero_grad()  # Clear gradients before each backward pass
        
                        g_task = self.forward(batch_xs)  # Get predictions
                        loss_task = nn.CrossEntropyLoss()(g_task, batch_ys)  # Compute loss

                        g_exemplar = self.forward(batch_ex_xs)
                        loss_exemplar = 1/p * nn.CrossEntropyLoss()(g_exemplar, batch_ex_ys)

                        loss = loss_task + loss_exemplar

                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update weights
            else:
                for epoch in range(num_epochs):
                    print(f"epoch is {epoch}")
                    for batch_xs, batch_ys in all_dataloader:
                        optimizer.zero_grad()  # Clear gradients before each backward pass
        
                        g_task = self.forward(batch_xs)  # Get predictions
                        loss_task = nn.CrossEntropyLoss()(g_task, batch_ys)  # Compute loss

                        loss = loss_task

                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update weights


        else:
            # icarl loss
            # access the old weights before training
            old_weights = self.state_dict()
            for epoch in range(num_epochs):
                for batch_xs, batch_ys in all_dataloader:
                    optimizer.zero_grad()
                    # let's compute q for this batch
                    # first save the current networks weight after gradient update
                    curr_weight = self.state_dict()
                    # now we set network weight to old weights
                    self.load_state_dict(old_weights)
                    # now we compute the forward function with old weights
                    q_batch = torch.sigmoid(self.forward(batch_xs)).detach()
                    # finally put back the current weights
                    self.load_state_dict(curr_weight)
                    g = torch.sigmoid(self.forward(batch_xs))
                    one_hot = F.one_hot(batch_ys, self.n_classes)
                    one_hot_labels = one_hot.clone().float()

                    loss_new = 0
                    loss_old = 0
                    criterion = nn.BCELoss()

                    for cls in range(self.n_known, self.n_classes):
                        loss_new += criterion(g[:,cls+1], one_hot_labels[:,cls])
                    
                    for cls in range(0, self.n_known):
                        loss_old += criterion(g[:,cls+1], q_batch[:,cls])
                    
                    loss_total = loss_new + loss_old
                    loss_total.backward()

                    optimizer.step()

        self.n_known += len(new_classes)

class iCaRL(MemorySetManager):
    def __init__(self,  p: float, n_classes: int, random_seed: int = 42):

        self.net = iCaRLNet(2048, 1) # create the iCaRLNet neural net object

        self.p = p
        self.random_seed = random_seed
        self.generator = torch.Generator().manual_seed(random_seed)
        self.memory_set_size = 0
        self.first_task = True

    def create_memory_set(self, x, y):
        """ Create or update memory set for new tasks """
        print(f"x.shape of incoming task data is {x.shape}, y.shape is {y.shape}")

        x_original = x # making a copy to select the right images

        transform_test = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # set the memory set size according to the size of the first task
        # MNIST has variable task sizes, so always use the first task size
        if self.first_task: 
            self.memory_set_size = int(self.p * len(x) / 2) # divide by two because we construct a coreset for each class
            print(f"memory set size is {self.memory_set_size}")

        if x.dim() == 2: # only reshape and duplicate color channel if MNIST
            x = x.view(-1, 28, 28)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        
        if x.dim() == 3:
            x = x.unsqueeze(1)

        print(f"\nReshaped x has size: {x.shape}")

        self.net.update_representation(x, y, self.p, "icarl")  # update the model with new data

        print("updated representation")

        classes = torch.unique(y)
        coreset_images = []
        coreset_labels = []

        for cls in classes:
            class_mask = (y == cls)
            class_images = x[class_mask]
            class_labels = y[class_mask]
            
            print(f"Starting selection for class {cls.item()}")
            coreset_indices = self.net.construct_exemplar_set(class_images, class_labels, self.memory_set_size, transform_test)
            
            coreset_images.append(x_original[coreset_indices])
            coreset_labels.append(y[coreset_indices])

        coreset = torch.cat(coreset_images)
        coreset_labels = torch.cat(coreset_labels)
        print(f"coreset after construction has shape: {coreset.shape}, {coreset_labels.shape}")
        print(f"\nmemory set after construction is: {self.net.exemplar_sets[-1].shape}")

        print("constructed the new memory set")
        
        self.first_task = False
        
        return coreset, coreset_labels



class GCRMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: fraction of task dataset to be included in replay buffer.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)
        np.random.seed(random_seed)
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 1.5
        self.lambda_val = 1  # need to figure out this hyperparameter

    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Initializes an empty memory replay buffer if training, called when task objects are created
        Else, use GCR to generate memory set

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        self.memory_set_size = int(x.shape[0] * self.p)
        if self.p == 1:
            return x, y
        
        # print(f"Shape of memory set in data.py is: {memory_x.shape}") # erik was just debugging some stuff
        print(f"Memory set size in data.py is: {self.memory_set_size}")
        return torch.empty(0), torch.empty(0)
