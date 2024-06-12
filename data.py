from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchDataset
from sklearn.cluster import KMeans

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

        print(f"Shape of memory set is: {memory_x.shape}") # erik was just debugging some stuff
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
num_epochs = 50
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
        self.fc = nn.Linear(feature_size, n_classes, bias=False)

        self.grayscale_to_rgb = transforms.Compose([transforms.Lambda(lambda x: torch.cat([x, x, x], dim=1))])


        self.n_classes = n_classes
        self.n_known = 0

        # list containing exemplar_sets
        # each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []
        self.exemplar_labels = []
        self.total_data = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        #self.optimizer = optim.SGD(self.parameters(), lr=2.0,
        #                           weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        # print(f"during forward, x shape is: {x.shape}")
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n
        

    def construct_exemplar_set(self, images, labels, m, transform):
        """Construct an exemplar set for image set

        Args:
            images: torch tensor containing images of a class
        """

        features = []
        self.feature_extractor.eval()
        with torch.no_grad():  # Ensures no gradients are calculated
            # for i, img in enumerate(images):
            for img in images:
                
                # print(f"before checking image dim: {img.shape}")
                if img.dim() == 3:  # Check if the channel dimension is missing
                    img = img.unsqueeze(0)  # Add a batch dimension if it's a single image
                img = img.to(DEVICE)
                
                if img.dim() == 1: # If it's mnist, it is just 784 flattened
                    img = img.view(1, 1, 28, 28) # add the color channel and reshape
                    img = self.grayscale_to_rgb(img)

                # print(f"img dimension is: {img.shape}")
                img = transform(img)  # Apply transformation

                # Extract features
                feature = self.feature_extractor(img).cpu().numpy()
                feature_norm = np.linalg.norm(feature)
                feature = feature / feature_norm  # Normalize
                features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean_norm = np.linalg.norm(class_mean)
        class_mean = class_mean / class_mean_norm # normalize

        exemplar_set = []
        exemplar_label = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        sum_inner = torch.zeros_like(class_mean).to(DEVICE) # base case for when there are no exemplars yet (k=0)

        for k in range(m):
            min_val = float('inf')
            arg_min_i = None

            for i, feature_x in enumerate(features): # this loop simulates the argmin process
  
                coefficient = 1/(k+1)
                sum_outer = feature_x + sum_inner
                sum_outer = coefficient * sum_outer

                normed_val = torch.linalg.norm(class_mean - sum_outer)

                if normed_val < min_val:
                    min_val = normed_val
                    arg_min_i = i

            exemplar_set.append(images[arg_min_i])
            exemplar_features.append(features[arg_min_i])
            exemplar_label.append(labels[arg_min_i])

            print(f"Shape of exemplar_features is: {torch.stack(exemplar_features).shape}")
            sum_inner = sum_inner.squeeze() + torch.stack(exemplar_features).sum(0)


    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]


    def combine_dataset_with_exemplars(self):
        for y, P_y in enumerate(self.exemplar_sets):
            self.total_data = [[x, y] for x, y in zip(self.exemplar_sets, self.exemplar_labels)]


    def update_representation(self, x, y):
        """Update the network representation using a tensor of images (x) and their corresponding labels (y)."""
        self.compute_means = True
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Identify and increment classes
        unique_classes = set(y.cpu().numpy().tolist())
        new_classes = [cls for cls in unique_classes if cls >= self.n_classes]
        self.increment_classes(len(new_classes))
        self.to(DEVICE)
        print(f"{len(new_classes)} new classes.")

        self.combine_dataset_with_exemplars()
        if self.total_data:
            exemplar_xs, exemplar_ys = zip(*self.total_data)

            # ensure all elements are tensors
            exemplar_xs = [torch.tensor(item, dtype=x.dtype, device=x.device) if not torch.is_tensor(item) else item for item in exemplar_xs]
            exemplar_ys = [torch.tensor(item, dtype=y.dtype, device=y.device) if not torch.is_tensor(item) else item for item in exemplar_ys]
            
            # stack tensors to maintain consistent dimensions for concatenation
            exemplar_xs = torch.stack(exemplar_xs)
            exemplar_ys = torch.stack(exemplar_ys)
        else:
            # initialize exemplar_xs and exemplar_ys as empty tensors with the remaining dimensions according to x and y
            exemplar_xs = torch.empty((0, *x.shape[1:]), dtype=x.dtype, device=x.device)
            exemplar_ys = torch.empty((0, *y.shape[1:]), dtype=y.dtype, device=y.device)
            

        # concatenate tensors
        print(f"exemplar_xs shape is {exemplar_xs.shape}, x shape is {x.shape}")
        print(f"exemplar_ys shape is {exemplar_ys.shape}, y shape during update is {y.shape}")

        if exemplar_xs.size(0) == 0 and exemplar_ys.size(0) == 0:

            print("exemplar start is empty (starting)")
            all_ys = torch.cat([exemplar_ys, y], dim=0)
            all_xs = torch.cat([exemplar_xs, x], dim=0) 
        else:
            print(f"exemplar set is not empty; it has size {exemplar_xs.shape}")
            num_images = exemplar_xs.size(0) * exemplar_xs.size(1) # the total number of images
            num_labels = exemplar_ys.size(0) * exemplar_ys.size(1)

            if exemplar_xs.size(-1) == 32:
                all_xs = torch.cat([exemplar_xs.reshape(num_images, 3, 32, 32), x], dim=0)
                all_ys = torch.cat([exemplar_ys.reshape(num_labels), y], dim=0)
            elif exemplar_xs.size(-1) == 784:
                all_xs = torch.cat([exemplar_xs.reshape(num_images, 784), x], dim=0)
                all_ys = torch.cat([exemplar_ys.reshape(num_labels), y], dim=0)


        combined_dataset = torch.utils.data.TensorDataset(all_xs, all_ys)
        loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # store network outputs with pre-update parameters
        q = torch.zeros(len(combined_dataset), self.n_classes)
        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):

                if images.dim() == 2:
                    images = images.view(-1, 1, 28, 28)
                    images = self.grayscale_to_rgb(images)
                    assert images.shape[1] == 3 # asserting that the images now have three color channels

                g = torch.sigmoid(self.forward(images))

                start_index = idx * loader.batch_size
                end_index = start_index + images.size(0)
                q[start_index:end_index] = g.data

        optimizer = self.optimizer

        # network training
        for epoch in range(num_epochs):
            print(f"epoch is {epoch}")
            for idx, (images, labels) in enumerate(loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                if images.dim() == 2:
                    images = images.view(-1, 1, 28, 28)
                    images = self.grayscale_to_rgb(images)
                    assert images.shape[1] == 3

                g = self.forward(images)

                # classification loss for new classes
                loss = self.cls_loss(g, labels)
                # distillation loss for old classes
                if self.n_known > 0:
                    g = torch.sigmoid(g)
                    q_i = q[idx]
                    dist_loss = sum(self.dist_loss(g[:, y], q_i[:, y]) for y in range(self.n_known))
                    loss += dist_loss

                loss.backward()
                optimizer.step()

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

        transform_test = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if self.first_task:
            self.memory_set_size = int(self.p * len(x))
            print(f"memory set size is {self.memory_set_size}")

        self.net.update_representation(x, y)  # update the model with new data

        print("updated memory sets")

        self.net.construct_exemplar_set(x, y, self.memory_set_size, transform_test)  # update the exemplar set for the new class
        # print(f"shape of memory set after construction is: {np.array(self.net.exemplar_sets).shape}")
        print(f"shape of memory set after construction is: {torch.tensor(self.net.exemplar_sets[-1]).shape}")
        print("constructed the new memory set")
        
        self.first_task = False

        return torch.tensor(self.net.exemplar_sets[-1]), torch.tensor(self.net.exemplar_labels[-1])
        # return self.net.exemplar_sets[-1], self.net.exemplar_labels[-1] # should return the last image set in the memory set
        # does I need to return a tensor?
        ## and their corresponding labels